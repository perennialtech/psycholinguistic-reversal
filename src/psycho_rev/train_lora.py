"""Scaled self-distillation + QLoRA SFT + SimPO preference optimization for psycholinguistic reversal.

Stage 0: Generate Best-of-N psycholinguistically verified rewrites from a teacher LLM on
         TRAIN_DENSE_CORPUS with incremental checkpointing and strict quality filtering.
Stage 1: Supervised Fine-Tuning (SFT) on (source -> chosen) + IDENTITY_REGULARIZERS with 90/10
         train/val split and validation loss logging.
Stage 2: Optional reference-free SimPO (Simple Preference Optimization, Meng et al. 2024)
         with an SFT anchor term to suppress semantic drift and hallucinations in <7 GB VRAM.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from psycho_rev.benchmark_data import IDENTITY_REGULARIZERS, TRAIN_DENSE_CORPUS
from psycho_rev.code_guard import protect_spans
from psycho_rev.features import PsycholinguisticAnalyzer
from psycho_rev.neural import (
    REWRITE_SYSTEM_PROMPT,
    BestOfNPsycholinguisticReverser,
    QwenZeroShotReverser,
)
from psycho_rev.semantics import SemanticEvaluator


def _encode_pair(
    src: str,
    tgt: str,
    tokenizer: AutoTokenizer,
    max_len: int = 512,
) -> dict[str, torch.Tensor]:
    g_src = protect_spans(src)
    g_tgt = tgt
    for tok, orig in g_src.placeholders.items():
        g_tgt = g_tgt.replace(orig, tok)

    prompt_msgs = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Rewrite this passage clearly:\n\n{g_src.masked}"},
    ]
    full_msgs = prompt_msgs + [{"role": "assistant", "content": g_tgt}]

    p_enc = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
    f_enc = tokenizer.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False)
    prompt_ids = list(p_enc["input_ids"]) if not isinstance(p_enc, list) else p_enc
    full_ids = list(f_enc["input_ids"]) if not isinstance(f_enc, list) else f_enc
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SFTDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tokenizer: AutoTokenizer, max_len: int = 512):
        self.items: list[dict[str, torch.Tensor]] = [
            _encode_pair(src, tgt, tokenizer, max_len=max_len) for src, tgt in pairs
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


class PreferenceDataset(Dataset):
    """Dataset of (source, chosen, rejected) triples for SimPO preference optimization."""

    def __init__(
        self,
        triples: list[tuple[str, str, str]],
        tokenizer: AutoTokenizer,
        max_len: int = 512,
    ):
        self.items: list[dict[str, torch.Tensor]] = []
        for src, chosen, rejected in triples:
            c_enc = _encode_pair(src, chosen, tokenizer, max_len=max_len)
            r_enc = _encode_pair(src, rejected, tokenizer, max_len=max_len)
            self.items.append(
                {
                    "chosen_input_ids": c_enc["input_ids"],
                    "chosen_labels": c_enc["labels"],
                    "rejected_input_ids": r_enc["input_ids"],
                    "rejected_labels": r_enc["labels"],
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


def _collate_sft(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_l = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), max_l), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_l), dtype=torch.long)
    labels = torch.full((len(batch), max_l), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[i, :length] = item["input_ids"]
        attention_mask[i, :length] = 1
        labels[i, :length] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _collate_pref(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    chosen = _collate_sft(
        [{"input_ids": x["chosen_input_ids"], "labels": x["chosen_labels"]} for x in batch],
        pad_id,
    )
    rejected = _collate_sft(
        [{"input_ids": x["rejected_input_ids"], "labels": x["rejected_labels"]} for x in batch],
        pad_id,
    )
    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "chosen_labels": chosen["labels"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
        "rejected_labels": rejected["labels"],
    }


def _sequence_avg_logp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute length-normalized log-probability over non-masked response tokens."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.clamp(min=0)
    token_logp = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    seq_sum = (token_logp * mask).sum(dim=-1)
    seq_len = mask.sum(dim=-1).clamp(min=1)
    return seq_sum / seq_len


def _eval_sft_loss(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    if len(loader) == 0:
        return 0.0
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total += model(**batch).loss.item()
    model.train()
    return total / len(loader)


def _eval_simpo_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    beta: float,
    gamma: float,
    sft_anchor_weight: float,
) -> tuple[float, float]:
    if len(loader) == 0:
        return 0.0, 0.0
    model.eval()
    total_loss = 0.0
    total_margin = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            c_logp = _sequence_avg_logp(
                model, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"]
            )
            r_logp = _sequence_avg_logp(
                model, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"]
            )
            margin = c_logp - r_logp
            simpo = -F.logsigmoid(beta * margin - gamma).mean()
            sft_anchor = -c_logp.mean()
            total_loss += (simpo + sft_anchor_weight * sft_anchor).item()
            total_margin += margin.mean().item()
    model.train()
    return total_loss / len(loader), total_margin / len(loader)


def build_distillation_pairs(
    out_pairs_path: Path,
    teacher_model: str = "Qwen/Qwen2.5-3B-Instruct",
    num_samples: int = 8,
    max_samples: int | None = None,
    resume: bool = False,
    checkpoint_every: int = 10,
    nli_model: str | None = None,
) -> list[dict[str, Any]]:
    """Generate and strictly filter Best-of-N teacher pairs with chosen/rejected preference data."""
    out_pairs_path = Path(out_pairs_path)
    out_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_pairs_path.with_suffix(".progress.jsonl")

    corpus = list(TRAIN_DENSE_CORPUS)
    if max_samples is not None and max_samples > 0:
        corpus = corpus[:max_samples]

    processed_indices: set[int] = set()
    kept_records: list[dict[str, Any]] = []

    if resume:
        if progress_path.exists():
            with open(progress_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    processed_indices.add(int(entry["idx"]))
                    if entry.get("kept") and entry.get("record"):
                        kept_records.append(entry["record"])
            print(
                f"Resuming distillation: {len(processed_indices)}/{len(corpus)} already processed, "
                f"{len(kept_records)} kept."
            )
        elif out_pairs_path.exists():
            with open(out_pairs_path) as f:
                kept_records = json.load(f)
            kept_sources = {r["source"] for r in kept_records}
            for i, s in enumerate(corpus):
                if s in kept_sources:
                    processed_indices.add(i)
            print(
                f"Resuming from {out_pairs_path}: {len(processed_indices)} matched sources, "
                f"{len(kept_records)} kept."
            )
    else:
        if progress_path.exists():
            progress_path.unlink()

    remaining = [(idx, src) for idx, src in enumerate(corpus) if idx not in processed_indices]
    if not remaining:
        with open(out_pairs_path, "w") as f:
            json.dump(kept_records, f, indent=2)
        return kept_records

    analyzer = PsycholinguisticAnalyzer()
    semantics = SemanticEvaluator(nli_model=nli_model) if nli_model else SemanticEvaluator()
    teacher = QwenZeroShotReverser(model_name=teacher_model)
    reranker = BestOfNPsycholinguisticReverser(teacher, analyzer, semantics, num_samples=num_samples)

    newly_processed = 0
    for idx, src in remaining:
        orig_feat = analyzer.analyze(src)
        ranked = reranker.rewrite_ranked_all(src)
        best = ranked[0]
        drop = orig_feat.difficulty_score - best.difficulty_score
        contra = float(best.nli_max_contradiction)
        lratio = float(best.length_ratio)

        keep = (
            drop >= 10.0
            and best.cosine_similarity >= 0.80
            and best.nli_min_entailment >= 0.72
            and contra <= 0.10
            and best.code_preservation >= 1.0
            and 0.70 <= lratio <= 1.60
        )

        record: dict[str, Any] | None = None
        if keep:
            rejected_cand = next((c for c in reversed(ranked[1:]) if c.text != best.text), None)
            rejected_text = rejected_cand.text if rejected_cand is not None else src
            rejected_obj = rejected_cand.objective if rejected_cand is not None else None
            record = {
                "id": f"train_{idx:03d}",
                "source": src,
                "target": best.text,
                "chosen": best.text,
                "rejected": rejected_text,
                "orig_difficulty": orig_feat.difficulty_score,
                "target_difficulty": best.difficulty_score,
                "difficulty_drop": round(drop, 2),
                "cosine_similarity": best.cosine_similarity,
                "nli_min_entailment": best.nli_min_entailment,
                "nli_max_contradiction": contra,
                "gpt2_perplexity": best.gpt2_perplexity,
                "code_preservation": best.code_preservation,
                "length_ratio": lratio,
                "rejected_objective": rejected_obj,
            }
            kept_records.append(record)
            print(
                f"[{idx + 1:03d}/{len(corpus)}] KEPT drop={drop:+.1f} "
                f"cos={best.cosine_similarity:.3f} nli={best.nli_min_entailment:.3f} "
                f"contra={contra:.3f} len={lratio:.2f}"
            )
        else:
            print(
                f"[{idx + 1:03d}/{len(corpus)}] SKIP drop={drop:+.1f} "
                f"cos={best.cosine_similarity:.3f} nli={best.nli_min_entailment:.3f} "
                f"contra={contra:.3f} len={lratio:.2f}"
            )

        with open(progress_path, "a") as pf:
            pf.write(json.dumps({"idx": idx, "kept": keep, "record": record}) + "\n")

        newly_processed += 1
        if newly_processed % checkpoint_every == 0:
            with open(out_pairs_path, "w") as f:
                json.dump(kept_records, f, indent=2)
            print(f"  [checkpoint] saved {len(kept_records)} kept pairs to {out_pairs_path}")

    teacher.close()
    with open(out_pairs_path, "w") as f:
        json.dump(kept_records, f, indent=2)
    return kept_records


def train_lora_adapter(
    pairs_path: Path,
    output_dir: Path,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    epochs: int = 3,
    dpo_epochs: int = 2,
    lr: float = 1.5e-4,
    dpo_lr: float = 5e-5,
    simpo_beta: float = 2.0,
    simpo_gamma: float = 0.5,
    sft_anchor_weight: float = 0.1,
    batch_size: int = 2,
    grad_accum: int = 4,
    use_4bit: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a LoRA/QLoRA adapter with Stage 1 SFT and optional Stage 2 SimPO preference optimization."""
    pairs_path = Path(pairs_path)
    output_dir = Path(output_dir)

    with open(pairs_path) as f:
        records = json.load(f)

    sft_pairs: list[tuple[str, str]] = [
        (r["source"], r.get("chosen", r["target"])) for r in records
    ]
    for h in IDENTITY_REGULARIZERS:
        sft_pairs.append((h, h))

    rng = random.Random(seed)
    rng.shuffle(sft_pairs)
    n_val = max(1, int(len(sft_pairs) * val_ratio)) if len(sft_pairs) >= 10 else 0
    val_pairs = sft_pairs[:n_val]
    train_pairs = sft_pairs[n_val:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device == "cuda" else torch.float32)
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=compute_dtype).to(device)
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = SFTDataset(train_pairs, tokenizer)
    val_ds = SFTDataset(val_pairs, tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate_sft(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate_sft(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    total_steps = max((len(train_loader) * epochs) // grad_accum, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(total_steps // 10, 1), num_training_steps=total_steps
    )

    model.train()
    t0 = time.time()
    sft_train_history: list[float] = []
    sft_val_history: list[float] = []

    print(
        f"=== Stage 1: SFT ({epochs} epochs, {len(train_pairs)} train / {len(val_pairs)} val, "
        f"4bit={use_4bit and device == 'cuda'}) ==="
    )
    for epoch in range(epochs):
        ep_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            ep_loss += out.loss.item()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        avg_train = ep_loss / max(len(train_loader), 1)
        avg_val = _eval_sft_loss(model, val_loader, device)
        sft_train_history.append(round(avg_train, 4))
        sft_val_history.append(round(avg_val, 4))
        print(f"SFT Epoch {epoch + 1}/{epochs} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

    # Stage 2: Reference-free SimPO preference optimization
    simpo_history: list[dict[str, float]] = []
    pref_triples: list[tuple[str, str, str]] = [
        (r["source"], r.get("chosen", r["target"]), r["rejected"])
        for r in records
        if r.get("rejected") and r["rejected"] != r.get("chosen", r["target"])
    ]
    if dpo_epochs > 0 and pref_triples:
        rng.shuffle(pref_triples)
        n_pref_val = max(1, int(len(pref_triples) * val_ratio)) if len(pref_triples) >= 10 else 0
        pref_val_ds = PreferenceDataset(pref_triples[:n_pref_val], tokenizer)
        pref_train_ds = PreferenceDataset(pref_triples[n_pref_val:], tokenizer)

        pref_train_loader = DataLoader(
            pref_train_ds,
            batch_size=max(1, batch_size // 2),
            shuffle=True,
            collate_fn=lambda b: _collate_pref(b, tokenizer.pad_token_id),
        )
        pref_val_loader = DataLoader(
            pref_val_ds,
            batch_size=max(1, batch_size // 2),
            shuffle=False,
            collate_fn=lambda b: _collate_pref(b, tokenizer.pad_token_id),
        )

        pref_opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=dpo_lr, weight_decay=0.01
        )
        pref_steps = max((len(pref_train_loader) * dpo_epochs) // grad_accum, 1)
        pref_sched = get_cosine_schedule_with_warmup(
            pref_opt, num_warmup_steps=max(pref_steps // 10, 1), num_training_steps=pref_steps
        )

        print(
            f"=== Stage 2: SimPO ({dpo_epochs} epochs, {len(pref_train_ds)} train / {len(pref_val_ds)} val, "
            f"beta={simpo_beta}, gamma={simpo_gamma}, sft_anchor={sft_anchor_weight}) ==="
        )
        model.train()
        for epoch in range(dpo_epochs):
            ep_loss = 0.0
            ep_margin = 0.0
            pref_opt.zero_grad()
            for step, batch in enumerate(pref_train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                c_logp = _sequence_avg_logp(
                    model, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"]
                )
                r_logp = _sequence_avg_logp(
                    model, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"]
                )
                margin = c_logp - r_logp
                simpo_loss = -F.logsigmoid(simpo_beta * margin - simpo_gamma).mean()
                sft_anchor = -c_logp.mean()
                total = simpo_loss + sft_anchor_weight * sft_anchor
                (total / grad_accum).backward()
                ep_loss += total.item()
                ep_margin += margin.mean().item()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(pref_train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    pref_opt.step()
                    pref_sched.step()
                    pref_opt.zero_grad()

            avg_loss = ep_loss / max(len(pref_train_loader), 1)
            avg_margin = ep_margin / max(len(pref_train_loader), 1)
            val_loss, val_margin = _eval_simpo_loss(
                model, pref_val_loader, device, simpo_beta, simpo_gamma, sft_anchor_weight
            )
            simpo_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(avg_loss, 4),
                    "train_margin": round(avg_margin, 4),
                    "val_loss": round(val_loss, 4),
                    "val_margin": round(val_margin, 4),
                }
            )
            print(
                f"SimPO Epoch {epoch + 1}/{dpo_epochs} | train_loss={avg_loss:.4f} "
                f"margin={avg_margin:+.4f} | val_loss={val_loss:.4f} val_margin={val_margin:+.4f}"
            )

    elapsed = round(time.time() - t0, 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta: dict[str, Any] = {
        "base_model": base_model,
        "use_4bit": use_4bit and device == "cuda",
        "num_distilled_pairs": len(records),
        "num_preference_triples": len(pref_triples),
        "num_identity_regularizers": len(IDENTITY_REGULARIZERS),
        "sft_epochs": epochs,
        "dpo_epochs": dpo_epochs,
        "lr": lr,
        "dpo_lr": dpo_lr,
        "simpo_beta": simpo_beta,
        "simpo_gamma": simpo_gamma,
        "sft_anchor_weight": sft_anchor_weight,
        "sft_train_loss_history": sft_train_history,
        "sft_val_loss_history": sft_val_history,
        "simpo_history": simpo_history,
        "train_seconds": elapsed,
    }
    with open(output_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m psycho_rev.train_lora",
        description="Distill Best-of-N teacher pairs and train a QLoRA/LoRA SFT + SimPO adapter.",
    )
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=Path("results/distilled_train_pairs.json"),
        help="Path to save/load distilled training pairs JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/qwen25-3b-psycho-rev-lora"),
        help="Directory to save the trained LoRA adapter.",
    )
    parser.add_argument(
        "--teacher-model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Teacher model ID for Best-of-N distillation.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base student model ID for LoRA fine-tuning.",
    )
    parser.add_argument("--num-samples", type=int, default=8, help="Best-of-N candidate count for distillation.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on TRAIN_DENSE_CORPUS passages.")
    parser.add_argument("--resume", action="store_true", help="Resume distillation from incremental checkpoint.")
    parser.add_argument("--distill-only", action="store_true", help="Run distillation only and skip LoRA training.")
    parser.add_argument("--skip-distill", action="store_true", help="Skip distillation and train from --pairs-path.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of Stage 1 SFT epochs.")
    parser.add_argument("--dpo-epochs", type=int, default=2, help="Number of Stage 2 SimPO epochs (0 to disable).")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate for Stage 1 SFT.")
    parser.add_argument("--dpo-lr", type=float, default=5e-5, help="Learning rate for Stage 2 SimPO.")
    parser.add_argument("--simpo-beta", type=float, default=2.0, help="SimPO beta scaling factor.")
    parser.add_argument("--simpo-gamma", type=float, default=0.5, help="SimPO target reward margin gamma.")
    parser.add_argument("--sft-anchor", type=float, default=0.1, help="Weight on SFT anchor loss during SimPO.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA and train in bf16/fp16.")
    parser.add_argument("--nli-model", default=None, help="Optional NLI cross-encoder model override.")
    args = parser.parse_args()

    if not args.skip_distill and (args.resume or args.distill_only or not args.pairs_path.exists()):
        build_distillation_pairs(
            out_pairs_path=args.pairs_path,
            teacher_model=args.teacher_model,
            num_samples=args.num_samples,
            max_samples=args.max_samples,
            resume=args.resume,
            nli_model=args.nli_model,
        )

    if args.distill_only:
        print(f"Distillation complete. Pairs saved to {args.pairs_path}")
        return

    meta = train_lora_adapter(
        pairs_path=args.pairs_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        dpo_epochs=args.dpo_epochs,
        lr=args.lr,
        dpo_lr=args.dpo_lr,
        simpo_beta=args.simpo_beta,
        simpo_gamma=args.simpo_gamma,
        sft_anchor_weight=args.sft_anchor,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        use_4bit=not args.no_4bit,
    )
    print("TRAINING COMPLETE:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
