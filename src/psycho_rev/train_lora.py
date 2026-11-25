"""Self-distillation + LoRA fine-tuning of Qwen2.5-1.5B-Instruct for psycholinguistic reversal.

Step 1: Generate Best-of-8 psycholinguistically verified rewrites from Qwen2.5-3B-Instruct
        on the disjoint training corpus (45 dense AI passages + identity-regularized human passages).
Step 2: Filter strictly for difficulty reduction + semantic fidelity + 100% code preservation.
Step 3: Fine-tune Qwen2.5-1.5B-Instruct with PEFT LoRA so a single greedy pass matches Best-of-N.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from psycho_rev.benchmark_data import TRAIN_DENSE_CORPUS
from psycho_rev.code_guard import protect_spans
from psycho_rev.features import PsycholinguisticAnalyzer
from psycho_rev.neural import (
    REWRITE_SYSTEM_PROMPT,
    BestOfNPsycholinguisticReverser,
    QwenZeroShotReverser,
)
from psycho_rev.semantics import SemanticEvaluator

# Short natural-prose identity regularizers (disjoint from HUMAN_CONTROL test set)
IDENTITY_REGULARIZERS: list[str] = [
    "We should finish the migration before Friday so the team has time to monitor the logs over the weekend.",
    "If `load_settings()` fails to open the file, it logs a warning and uses the default configuration.",
    "The bus arrives every fifteen minutes during the morning rush, which makes commuting downtown much easier.",
    "Clear documentation saves everyone time because new contributors can find answers without asking in chat.",
    "After you update `config/routes.yaml`, restart the service and check that the health endpoint returns 200.",
    "Most readers understand short, direct sentences faster than long clauses packed with technical jargon.",
]


class SFTDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tokenizer: AutoTokenizer, max_len: int = 512):
        self.items: list[dict[str, torch.Tensor]] = []
        for src, tgt in pairs:
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
            self.items.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


def _collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_l = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), max_l), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_l), dtype=torch.long)
    labels = torch.full((len(batch), max_l), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        l = len(item["input_ids"])
        input_ids[i, :l] = item["input_ids"]
        attention_mask[i, :l] = 1
        labels[i, :l] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_distillation_pairs(
    out_pairs_path: Path,
    teacher_model: str = "Qwen/Qwen2.5-3B-Instruct",
    num_samples: int = 8,
) -> list[dict]:
    analyzer = PsycholinguisticAnalyzer()
    semantics = SemanticEvaluator()
    teacher = QwenZeroShotReverser(model_name=teacher_model)
    reranker = BestOfNPsycholinguisticReverser(teacher, analyzer, semantics, num_samples=num_samples)

    kept_records: list[dict] = []
    for idx, src in enumerate(TRAIN_DENSE_CORPUS):
        orig_feat = analyzer.analyze(src)
        best = reranker.rewrite_ranked(src)
        drop = orig_feat.difficulty_score - best.difficulty_score
        if drop >= 5.0 and best.cosine_similarity >= 0.78 and best.nli_min_entailment >= 0.50 and best.code_preservation >= 1.0:
            kept_records.append(
                {
                    "id": f"train_{idx:02d}",
                    "source": src,
                    "target": best.text,
                    "orig_difficulty": orig_feat.difficulty_score,
                    "target_difficulty": best.difficulty_score,
                    "difficulty_drop": round(drop, 2),
                    "cosine_similarity": best.cosine_similarity,
                    "nli_min_entailment": best.nli_min_entailment,
                }
            )
            print(
                f"[{idx+1:02d}/{len(TRAIN_DENSE_CORPUS)}] KEPT drop={drop:+.1f} "
                f"cos={best.cosine_similarity:.3f} nli={best.nli_min_entailment:.3f}"
            )
        else:
            print(
                f"[{idx+1:02d}/{len(TRAIN_DENSE_CORPUS)}] SKIP drop={drop:+.1f} "
                f"cos={best.cosine_similarity:.3f} nli={best.nli_min_entailment:.3f}"
            )

    teacher.close()
    out_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pairs_path, "w") as f:
        json.dump(kept_records, f, indent=2)
    return kept_records


def train_lora_adapter(
    pairs_path: Path,
    output_dir: Path,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    epochs: int = 3,
    lr: float = 1.5e-4,
    batch_size: int = 2,
    grad_accum: int = 4,
) -> dict:
    with open(pairs_path) as f:
        records = json.load(f)

    pairs = [(r["source"], r["target"]) for r in records]
    for h in IDENTITY_REGULARIZERS:
        pairs.append((h, h))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
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

    dataset = SFTDataset(pairs, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max((len(loader) * epochs) // grad_accum, 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(total_steps // 10, 1), num_training_steps=total_steps)

    model.train()
    t0 = time.time()
    loss_history: list[float] = []
    for epoch in range(epochs):
        ep_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            ep_loss += out.loss.item()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        avg_loss = ep_loss / max(len(loader), 1)
        loss_history.append(round(avg_loss, 4))
        print(f"Epoch {epoch+1}/{epochs} | train_loss={avg_loss:.4f}")

    elapsed = round(time.time() - t0, 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "base_model": base_model,
        "num_distilled_pairs": len(records),
        "num_identity_regularizers": len(IDENTITY_REGULARIZERS),
        "epochs": epochs,
        "lr": lr,
        "loss_history": loss_history,
        "train_seconds": elapsed,
    }
    with open(output_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


if __name__ == "__main__":
    pairs_file = Path("results/distilled_train_pairs.json")
    adapter_out = Path("checkpoints/qwen25-1.5b-psycho-rev-lora")
    if not pairs_file.exists():
        build_distillation_pairs(pairs_file)
    meta = train_lora_adapter(pairs_file, adapter_out)
    print("TRAINING COMPLETE:", json.dumps(meta, indent=2))
