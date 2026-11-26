"""Scaled teacher-distillation + QLoRA SFT + SimPO preference optimization for psycholinguistic reversal.

Stage 0: Optionally generate Best-of-N psycholinguistically verified rewrites from a teacher LLM on
         TRAIN_DENSE_CORPUS with incremental checkpointing, OR load an external distilled JSON/JSONL
         dataset via --dataset-path.
Stage 1: Strict quality filtering (exact code preservation, length bounds, non-truncation, optional
         NLI/acceptability thresholds, balanced identity no-op ratio), deterministic train/val split,
         and Supervised Fine-Tuning (SFT) with per-epoch validation loss and best-checkpoint selection.
Stage 2: Optional reference-free SimPO (Simple Preference Optimization, Meng et al. 2024) on hard
         negatives with length-normalized log-probabilities, an SFT anchor term, 4-bit NF4 QLoRA,
         gradient checkpointing, and best-checkpoint selection for 10 GB VRAM GPUs.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import shutil
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from psycho_rev.benchmark_data import IDENTITY_REGULARIZERS, TRAIN_DENSE_CORPUS
from psycho_rev.code_guard import code_preservation_rate, protect_spans
from psycho_rev.features import PsycholinguisticAnalyzer
from psycho_rev.neural import (
    REWRITE_SYSTEM_PROMPT,
    BestOfNPsycholinguisticReverser,
    QwenZeroShotReverser,
)
from psycho_rev.semantics import SemanticEvaluator

logger = logging.getLogger(__name__)

_TERMINAL_PUNCT = (".", "!", "?", '"', "'", ")", "]", "`")
_DANGLING_PUNCT = (",", ";", ":", "(", "[", "{")


def load_distilled_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load distilled training records from a JSONL or JSON file.

    Supports:
      - `.jsonl` / newline-delimited JSON objects (skipping blank lines)
      - `.json` containing either a list of record dicts or a dict with a
        ``records``, ``pairs``, or ``data`` list.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    raw_text = dataset_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    records: list[dict[str, Any]] = []
    if dataset_path.suffix.lower() == ".jsonl" or (
        not raw_text.startswith("[") and not raw_text.startswith("{")
    ):
        for lineno, line in enumerate(raw_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {lineno} in {dataset_path} is not a JSON object.")
            records.append(obj)
    else:
        # Handle JSON files as well as multi-line JSONL files without .jsonl extension
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            for lineno, line in enumerate(raw_text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                if not isinstance(obj, dict):
                    raise ValueError(f"Line {lineno} in {dataset_path} is not a JSON object.")
                records.append(obj)
        else:
            if isinstance(parsed, list):
                records = [r for r in parsed if isinstance(r, dict)]
            elif isinstance(parsed, dict):
                for key in ("records", "pairs", "data"):
                    if isinstance(parsed.get(key), list):
                        records = [r for r in parsed[key] if isinstance(r, dict)]
                        break
                else:
                    raise ValueError(
                        f"JSON object in {dataset_path} must contain a 'records', 'pairs', or 'data' list."
                    )
            else:
                raise ValueError(f"Unsupported JSON root type in {dataset_path}: {type(parsed).__name__}")

    normalized: list[dict[str, Any]] = []
    for rec in records:
        norm = dict(rec)
        src = str(norm.get("source", "")).strip()
        tgt_raw = norm.get("chosen") if norm.get("chosen") is not None else norm.get("target", "")
        tgt = str(tgt_raw).strip()
        norm["source"] = src
        norm["target"] = tgt
        norm["chosen"] = tgt
        if norm.get("rejected") is not None:
            norm["rejected"] = str(norm["rejected"]).strip()
        if "is_identity" not in norm or norm["is_identity"] is None:
            norm["is_identity"] = bool(src) and tgt == src
        else:
            norm["is_identity"] = bool(norm["is_identity"])
        if "domain" not in norm or norm["domain"] is None:
            norm["domain"] = "unknown"
        normalized.append(norm)
    return normalized


def compute_length_ratio(source: str, target: str) -> float:
    """Compute target-to-source word-count length ratio."""
    src_words = len(source.split())
    tgt_words = len(target.split())
    if src_words == 0:
        return 0.0 if tgt_words == 0 else float("inf")
    return tgt_words / src_words


def is_truncated_text(source: str, target: str, record: dict[str, Any] | None = None) -> bool:
    """Detect empty or truncated outputs from metadata flags or structural heuristics."""
    src = source.strip()
    tgt = target.strip()
    if not src or not tgt:
        return True

    if record is not None:
        if record.get("truncated") is True:
            return True
        if str(record.get("finish_reason", "")).lower() == "length":
            return True

    # Unbalanced fenced code blocks indicate cut-off generation
    if tgt.count("```") % 2 != 0:
        return True

    # Unexpected trailing ellipsis not present in the source
    if (tgt.endswith("...") or tgt.endswith("\u2026")) and not (
        src.endswith("...") or src.endswith("\u2026")
    ):
        return True

    # Dangling clause-opening/continuation punctuation on multi-word outputs when source was complete
    if len(tgt.split()) >= 6 and src.endswith(_TERMINAL_PUNCT) and tgt.endswith(_DANGLING_PUNCT):
        return True

    return False


def _first_numeric(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        val = record.get(key)
        if val is not None:
            try:
                num = float(val)
                if not math.isnan(num):
                    return num
            except (TypeError, ValueError):
                continue
    return None


def filter_distilled_records(
    records: list[dict[str, Any]],
    *,
    min_code_preservation: float = 1.0,
    min_length_ratio: float = 0.65,
    max_length_ratio: float = 1.85,
    min_nli_entailment: float = 0.70,
    max_nli_contradiction: float = 0.15,
    min_cosine_similarity: float = 0.75,
    min_acceptability: float = 0.60,
    identity_ratio: float = 0.15,
    include_builtin_identity: bool = False,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Strictly filter distilled pairs and balance identity (no-op) examples.

    Returns ``(filtered_records, filter_stats)``.
    """
    stats: dict[str, Any] = {
        "input_records": len(records),
        "rejected_empty_or_truncated": 0,
        "rejected_code_preservation": 0,
        "rejected_length_ratio": 0,
        "rejected_nli_entailment": 0,
        "rejected_nli_contradiction": 0,
        "rejected_cosine_similarity": 0,
        "rejected_acceptability": 0,
        "rejected_duplicate": 0,
        "identity_capped": 0,
        "builtin_identity_added": 0,
    }

    rewrite_kept: list[dict[str, Any]] = []
    identity_kept: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for rec in records:
        src = str(rec.get("source", "")).strip()
        tgt_raw = rec.get("chosen") if rec.get("chosen") is not None else rec.get("target", "")
        tgt = str(tgt_raw).strip()

        if is_truncated_text(src, tgt, rec):
            stats["rejected_empty_or_truncated"] += 1
            continue

        pair_key = (src, tgt)
        if pair_key in seen_pairs:
            stats["rejected_duplicate"] += 1
            continue

        is_identity = bool(rec.get("is_identity")) or (tgt == src)
        if is_identity and tgt != src:
            # Explicit is_identity=True must have target == source
            is_identity = False

        # Exact code/URL span preservation check
        computed_cp = code_preservation_rate(src, tgt)
        meta_cp = _first_numeric(rec, ("code_preservation", "code_preservation_rate"))
        effective_cp = min(computed_cp, meta_cp) if meta_cp is not None else computed_cp
        if effective_cp < min_code_preservation:
            stats["rejected_code_preservation"] += 1
            continue

        # Length ratio check
        lratio = compute_length_ratio(src, tgt)
        if not (min_length_ratio <= lratio <= max_length_ratio):
            stats["rejected_length_ratio"] += 1
            continue

        # Optional semantic / NLI / acceptability thresholds when present in metadata
        nli_ent = _first_numeric(rec, ("nli_min_entailment", "nli_entailment"))
        if nli_ent is not None and nli_ent < min_nli_entailment:
            stats["rejected_nli_entailment"] += 1
            continue

        nli_contra = _first_numeric(rec, ("nli_max_contradiction", "nli_contradiction"))
        if nli_contra is not None and nli_contra > max_nli_contradiction:
            stats["rejected_nli_contradiction"] += 1
            continue

        cos_sim = _first_numeric(rec, ("cosine_similarity",))
        if cos_sim is not None and cos_sim < min_cosine_similarity:
            stats["rejected_cosine_similarity"] += 1
            continue

        accept = _first_numeric(rec, ("acceptability_min", "acceptability_score", "acceptability", "cola_score"))
        if accept is not None and accept < min_acceptability:
            stats["rejected_acceptability"] += 1
            continue

        seen_pairs.add(pair_key)
        clean_rec = dict(rec)
        clean_rec["source"] = src
        clean_rec["target"] = tgt
        clean_rec["chosen"] = tgt
        clean_rec["is_identity"] = is_identity
        clean_rec["code_preservation"] = round(effective_cp, 4)
        clean_rec["length_ratio"] = round(lratio, 4)
        if "domain" not in clean_rec or clean_rec["domain"] is None:
            clean_rec["domain"] = "identity" if is_identity else "unknown"

        if is_identity:
            identity_kept.append(clean_rec)
        else:
            rewrite_kept.append(clean_rec)

    if include_builtin_identity and identity_ratio > 0.0:
        for idx, passage in enumerate(IDENTITY_REGULARIZERS):
            text = passage.strip()
            if not text or (text, text) in seen_pairs:
                continue
            seen_pairs.add((text, text))
            identity_kept.append(
                {
                    "id": f"builtin_identity_{idx:03d}",
                    "source": text,
                    "target": text,
                    "chosen": text,
                    "is_identity": True,
                    "domain": "identity",
                    "code_preservation": 1.0,
                    "length_ratio": 1.0,
                }
            )
            stats["builtin_identity_added"] += 1

    # Cap identity examples to at most identity_ratio of the final dataset
    rng = random.Random(seed)
    if identity_ratio <= 0.0:
        stats["identity_capped"] = len(identity_kept)
        identity_selected: list[dict[str, Any]] = []
    elif identity_ratio >= 1.0 or not rewrite_kept:
        identity_selected = list(identity_kept)
    else:
        max_identity = int(math.floor(len(rewrite_kept) * identity_ratio / (1.0 - identity_ratio)))
        if len(identity_kept) > max_identity:
            shuffled_id = list(identity_kept)
            rng.shuffle(shuffled_id)
            identity_selected = shuffled_id[:max_identity]
            stats["identity_capped"] = len(identity_kept) - max_identity
        else:
            identity_selected = list(identity_kept)

    combined = rewrite_kept + identity_selected
    stats["kept_rewrite_records"] = len(rewrite_kept)
    stats["kept_identity_records"] = len(identity_selected)
    stats["kept_total_records"] = len(combined)
    stats["effective_identity_ratio"] = (
        round(len(identity_selected) / len(combined), 4) if combined else 0.0
    )
    return combined, stats


def split_train_val(
    records: list[dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split records into (train, val), stratifying by (is_identity, domain)."""
    if not records or val_ratio <= 0.0 or len(records) < 2:
        return list(records), []

    rng = random.Random(seed)
    target_val = max(1, int(round(len(records) * val_ratio)))
    target_val = min(target_val, len(records) - 1)

    buckets: dict[tuple[bool, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = (bool(rec.get("is_identity", False)), str(rec.get("domain", "unknown")))
        buckets[key].append(rec)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []

    if len(buckets) > 1 and len(records) >= 10:
        for key in sorted(buckets.keys()):
            group = list(buckets[key])
            rng.shuffle(group)
            n_val_group = int(round(len(group) * val_ratio))
            if len(group) >= 2 and n_val_group >= len(group):
                n_val_group = len(group) - 1
            val_records.extend(group[:n_val_group])
            train_records.extend(group[n_val_group:])

        # Rebalance to hit target_val cleanly if rounding drifted
        rng.shuffle(train_records)
        rng.shuffle(val_records)
        while len(val_records) < target_val and len(train_records) > 1:
            val_records.append(train_records.pop())
        while len(val_records) > target_val:
            train_records.append(val_records.pop())
    else:
        shuffled = list(records)
        rng.shuffle(shuffled)
        val_records = shuffled[:target_val]
        train_records = shuffled[target_val:]

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def build_preference_triples(
    records: list[dict[str, Any]],
    fallback_to_source: bool = True,
) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    """Extract (source, chosen, rejected) triples for SimPO preference optimization.

    Prefers explicit hard-negative ``rejected`` candidates from the distilled dataset,
    falling back to ``source`` only when ``rejected`` is absent (and ``chosen != source``).
    """
    triples: list[tuple[str, str, str]] = []
    counts = {"hard_negatives": 0, "source_fallback": 0, "skipped": 0}

    for rec in records:
        src = str(rec.get("source", "")).strip()
        chosen = str(rec.get("chosen") if rec.get("chosen") is not None else rec.get("target", "")).strip()
        rejected_raw = rec.get("rejected")
        rejected = str(rejected_raw).strip() if rejected_raw is not None else ""

        if not src or not chosen:
            counts["skipped"] += 1
            continue

        if rejected and rejected != chosen:
            triples.append((src, chosen, rejected))
            counts["hard_negatives"] += 1
        elif fallback_to_source and src != chosen:
            triples.append((src, chosen, src))
            counts["source_fallback"] += 1
        else:
            counts["skipped"] += 1

    return triples, counts


def format_rewrite_messages(src: str, tgt: str | None = None) -> list[dict[str, str]]:
    """Build chat messages using REWRITE_SYSTEM_PROMPT and code_guard placeholders.

    Produces the exact same system + user prompt format used at inference in
    ``psycho_rev.neural.LoRASinglePassReverser`` and ``QwenZeroShotReverser``.
    """
    g_src = protect_spans(src)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Rewrite this passage clearly:\n\n{g_src.masked}"},
    ]
    if tgt is not None:
        if tgt == src:
            g_tgt = g_src.masked
        else:
            g_tgt = tgt
            # Replace longer originals first to avoid partial substring collisions,
            # preserving occurrence order for duplicate spans with count=1 so each
            # occurrence maps to its distinct CODEITEM<k> token.
            ordered_spans = sorted(
                g_src.placeholders.items(),
                key=lambda kv: -len(kv[1]),
            )
            for tok, orig in ordered_spans:
                g_tgt = g_tgt.replace(orig, tok, 1)
        messages.append({"role": "assistant", "content": g_tgt})
    return messages


def _split_paragraphs(text: str) -> list[str]:
    """Split a document into non-empty paragraphs separated by blank lines."""
    return [para.strip() for para in text.strip().split("\n\n") if para.strip()]


def encode_pair(
    src: str,
    tgt: str,
    tokenizer: Any,
    max_len: int = 1024,
) -> dict[str, torch.Tensor] | None:
    """Tokenize a (source, target) pair with prompt tokens masked to -100 in labels.

    Returns ``None`` if the full sequence exceeds ``max_len`` (never slicing mid-sentence
    or dropping the closing EOS token) or if no response tokens are present.
    """
    prompt_msgs = format_rewrite_messages(src)
    full_msgs = format_rewrite_messages(src, tgt)

    p_enc = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
    f_enc = tokenizer.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False)
    prompt_ids = list(p_enc["input_ids"]) if not isinstance(p_enc, list) else list(p_enc)
    full_ids = list(f_enc["input_ids"]) if not isinstance(f_enc, list) else list(f_enc)

    if len(full_ids) > max_len or len(full_ids) <= len(prompt_ids):
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


_encode_pair = encode_pair


class SFTDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tokenizer: Any, max_len: int = 1024):
        self.items: list[dict[str, torch.Tensor]] = []
        self.dropped_truncated: int = 0
        self.split_multipara_docs: int = 0
        for src, tgt in pairs:
            encoded = encode_pair(src, tgt, tokenizer, max_len=max_len)
            if encoded is not None:
                self.items.append(encoded)
                continue

            src_paras = _split_paragraphs(src)
            tgt_paras = _split_paragraphs(tgt)
            if len(src_paras) > 1 and len(src_paras) == len(tgt_paras):
                kept_any = False
                for s_para, t_para in zip(src_paras, tgt_paras):
                    sub_enc = encode_pair(s_para, t_para, tokenizer, max_len=max_len)
                    if sub_enc is not None:
                        self.items.append(sub_enc)
                        kept_any = True
                    else:
                        self.dropped_truncated += 1
                if kept_any:
                    self.split_multipara_docs += 1
            else:
                self.dropped_truncated += 1

        if self.dropped_truncated > 0:
            logger.warning(
                "SFTDataset dropped %d oversized example(s) exceeding max_len=%d (split %d multi-paragraph doc(s)).",
                self.dropped_truncated,
                max_len,
                self.split_multipara_docs,
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


class PreferenceDataset(Dataset):
    """Dataset of (source, chosen, rejected) triples for SimPO preference optimization."""

    def __init__(
        self,
        triples: list[tuple[str, str, str]],
        tokenizer: Any,
        max_len: int = 1024,
    ):
        self.items: list[dict[str, torch.Tensor]] = []
        self.dropped_truncated: int = 0
        self.split_multipara_docs: int = 0
        for src, chosen, rejected in triples:
            c_enc = encode_pair(src, chosen, tokenizer, max_len=max_len)
            r_enc = encode_pair(src, rejected, tokenizer, max_len=max_len)
            if c_enc is not None and r_enc is not None:
                self.items.append(
                    {
                        "chosen_input_ids": c_enc["input_ids"],
                        "chosen_labels": c_enc["labels"],
                        "rejected_input_ids": r_enc["input_ids"],
                        "rejected_labels": r_enc["labels"],
                    }
                )
                continue

            src_paras = _split_paragraphs(src)
            chosen_paras = _split_paragraphs(chosen)
            rejected_paras = _split_paragraphs(rejected)
            if (
                len(src_paras) > 1
                and len(src_paras) == len(chosen_paras) == len(rejected_paras)
            ):
                kept_any = False
                dropped_sub = 0
                for s_para, c_para, r_para in zip(src_paras, chosen_paras, rejected_paras):
                    if c_para == r_para:
                        continue
                    sc_enc = encode_pair(s_para, c_para, tokenizer, max_len=max_len)
                    sr_enc = encode_pair(s_para, r_para, tokenizer, max_len=max_len)
                    if sc_enc is not None and sr_enc is not None:
                        self.items.append(
                            {
                                "chosen_input_ids": sc_enc["input_ids"],
                                "chosen_labels": sc_enc["labels"],
                                "rejected_input_ids": sr_enc["input_ids"],
                                "rejected_labels": sr_enc["labels"],
                            }
                        )
                        kept_any = True
                    else:
                        dropped_sub += 1
                if kept_any:
                    self.split_multipara_docs += 1
                    self.dropped_truncated += dropped_sub
                else:
                    self.dropped_truncated += max(dropped_sub, 1)
            else:
                self.dropped_truncated += 1

        if self.dropped_truncated > 0:
            logger.warning(
                "PreferenceDataset dropped %d oversized triple(s) exceeding max_len=%d (split %d multi-paragraph doc(s)).",
                self.dropped_truncated,
                max_len,
                self.split_multipara_docs,
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


def sequence_avg_logp_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute length-normalized log-probability over non-masked response tokens from logits.

    Uses fused ``F.cross_entropy`` rather than materializing a full ``[B, T, V]`` log-softmax
    tensor, cutting peak logit memory roughly in half for large-vocabulary causal LMs.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    vocab_size = shift_logits.shape[-1]
    per_token_nll = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    mask = shift_labels != -100
    seq_sum = -(per_token_nll * mask).sum(dim=-1)
    seq_len = mask.sum(dim=-1).clamp(min=1)
    return seq_sum / seq_len


def _sequence_avg_logp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute length-normalized log-probability over non-masked response tokens."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return sequence_avg_logp_from_logits(logits, labels)


def compute_simpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
    sft_anchor_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute reference-free SimPO loss with optional SFT anchor term.

    Args:
        chosen_logp: Length-normalized average log-probability of chosen responses ``[B]``.
        rejected_logp: Length-normalized average log-probability of rejected responses ``[B]``.
        beta: Reward scaling factor.
        gamma: Target reward margin.
        sft_anchor_weight: Weight on negative chosen avg log-prob regularizer.
    """
    margin = chosen_logp - rejected_logp
    simpo_loss = -F.logsigmoid(beta * margin - gamma).mean()
    sft_anchor = -chosen_logp.mean()
    total_loss = simpo_loss + sft_anchor_weight * sft_anchor
    pref_accuracy = (margin > 0).float().mean()
    return {
        "loss": total_loss,
        "simpo_loss": simpo_loss,
        "sft_anchor": sft_anchor,
        "margin": margin.mean(),
        "accuracy": pref_accuracy,
    }


def _autocast_ctx(device: str, compute_dtype: torch.dtype | None):
    if device == "cuda" and compute_dtype is not None:
        return torch.amp.autocast(device_type="cuda", dtype=compute_dtype)
    return nullcontext()


def _eval_sft_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    compute_dtype: torch.dtype | None = None,
) -> float:
    if len(loader) == 0:
        return 0.0
    model.eval()
    total = 0.0
    with torch.inference_mode(), _autocast_ctx(device, compute_dtype):
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
    compute_dtype: torch.dtype | None = None,
) -> tuple[float, float, float]:
    if len(loader) == 0:
        return 0.0, 0.0, 0.0
    model.eval()
    total_loss = 0.0
    total_margin = 0.0
    total_acc = 0.0
    with torch.inference_mode(), _autocast_ctx(device, compute_dtype):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            c_logp = _sequence_avg_logp(
                model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            r_logp = _sequence_avg_logp(
                model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )
            metrics = compute_simpo_loss(
                c_logp,
                r_logp,
                beta=beta,
                gamma=gamma,
                sft_anchor_weight=sft_anchor_weight,
            )
            total_loss += metrics["loss"].item()
            total_margin += metrics["margin"].item()
            total_acc += metrics["accuracy"].item()
    model.train()
    n = len(loader)
    return total_loss / n, total_margin / n, total_acc / n


def _create_optimizer(
    params: list[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    use_paged_8bit: bool,
) -> torch.optim.Optimizer:
    if use_paged_8bit:
        try:
            import bitsandbytes as bnb

            return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except Exception:
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _load_adapter_weights_into_model(model: PeftModel, checkpoint_dir: Path) -> None:
    """Reload saved LoRA adapter weights from checkpoint_dir into an active PeftModel."""
    from peft import set_peft_model_state_dict
    from peft.utils import load_peft_weights

    adapter_state = load_peft_weights(str(checkpoint_dir))
    set_peft_model_state_dict(model, adapter_state)


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
            with open(progress_path, encoding="utf-8") as f:
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
            kept_records = load_distilled_dataset(out_pairs_path)
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
        with open(out_pairs_path, "w", encoding="utf-8") as f:
            json.dump(kept_records, f, indent=2)
        return kept_records

    analyzer = PsycholinguisticAnalyzer()
    semantics = SemanticEvaluator(nli_model=nli_model) if nli_model else SemanticEvaluator()
    teacher: QwenZeroShotReverser | None = None
    try:
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
                and 0.65 <= lratio <= 1.85
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
                    "is_identity": False,
                    "domain": "self_distilled",
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

            with open(progress_path, "a", encoding="utf-8") as pf:
                pf.write(json.dumps({"idx": idx, "kept": keep, "record": record}) + "\n")

            newly_processed += 1
            if newly_processed % checkpoint_every == 0:
                with open(out_pairs_path, "w", encoding="utf-8") as f:
                    json.dump(kept_records, f, indent=2)
                print(f"  [checkpoint] saved {len(kept_records)} kept pairs to {out_pairs_path}")

    finally:
        if teacher is not None:
            teacher.close()
        semantics.close()
    with open(out_pairs_path, "w", encoding="utf-8") as f:
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
    simpo_batch_size: int = 1,
    grad_accum: int = 4,
    max_length: int = 1024,
    use_4bit: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
    identity_ratio: float = 0.15,
    include_builtin_identity: bool = True,
    min_length_ratio: float = 0.65,
    max_length_ratio: float = 1.85,
    min_nli_entailment: float = 0.70,
    max_nli_contradiction: float = 0.15,
    min_cosine_similarity: float = 0.75,
    min_acceptability: float = 0.60,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    resume_from_checkpoint: Path | None = None,
    push_to_hub: str | None = None,
    hub_private: bool = False,
) -> dict[str, Any]:
    """Train a QLoRA/LoRA adapter with strict filtering, best-checkpoint selection, SFT, and SimPO."""
    pairs_path = Path(pairs_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    raw_records = load_distilled_dataset(pairs_path)
    filtered_records, filter_stats = filter_distilled_records(
        raw_records,
        min_code_preservation=1.0,
        min_length_ratio=min_length_ratio,
        max_length_ratio=max_length_ratio,
        min_nli_entailment=min_nli_entailment,
        max_nli_contradiction=max_nli_contradiction,
        min_cosine_similarity=min_cosine_similarity,
        min_acceptability=min_acceptability,
        identity_ratio=identity_ratio,
        include_builtin_identity=include_builtin_identity,
        seed=seed,
    )
    if not filtered_records:
        raise ValueError(
            f"No valid training pairs remained after quality filtering on {pairs_path}. "
            f"Filter stats: {json.dumps(filter_stats)}"
        )

    train_records, val_records = split_train_val(filtered_records, val_ratio=val_ratio, seed=seed)

    sft_train_pairs = [(r["source"], r["chosen"]) for r in train_records]
    sft_val_pairs = [(r["source"], r["chosen"]) for r in val_records]

    train_pref_triples, train_pref_stats = build_preference_triples(train_records, fallback_to_source=True)
    val_pref_triples, val_pref_stats = build_preference_triples(val_records, fallback_to_source=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_qlora = bool(use_4bit and device == "cuda")
    compute_dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device == "cuda" else torch.float32)
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=compute_dtype).to(device)
        try:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()

    base.config.use_cache = False

    if resume_from_checkpoint is not None:
        ckpt_path = Path(resume_from_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint directory not found: {ckpt_path}")
        model = PeftModel.from_pretrained(base, str(ckpt_path), is_trainable=True)
        print(f"Resumed trainable LoRA adapter from {ckpt_path}")
    else:
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    sft_best_dir = output_dir / "stage1_sft_best"
    simpo_best_dir = output_dir / "stage2_simpo_best"
    final_best_source: Path | None = None

    t0 = time.time()
    sft_history: list[dict[str, Any]] = []
    best_sft_loss = float("inf")
    best_sft_epoch = 0

    if epochs > 0:
        train_ds = SFTDataset(sft_train_pairs, tokenizer, max_len=max_length)
        val_ds = SFTDataset(sft_val_pairs, tokenizer, max_len=max_length)
        if len(train_ds) == 0:
            raise ValueError("SFT training dataset is empty after tokenization/truncation.")

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

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = _create_optimizer(trainable_params, lr=lr, weight_decay=0.01, use_paged_8bit=use_qlora)
        total_steps = max(math.ceil(len(train_loader) / grad_accum) * epochs, 1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=max(total_steps // 10, 1), num_training_steps=total_steps
        )

        print(
            f"=== Stage 1: SFT ({epochs} epochs, {len(train_ds)} train / {len(val_ds)} val, "
            f"max_len={max_length}, 4bit={use_qlora}) ==="
        )
        model.train()
        for epoch in range(epochs):
            ep_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with _autocast_ctx(device, compute_dtype):
                    out = model(**batch)
                    loss = out.loss / grad_accum
                loss.backward()
                ep_loss += out.loss.item()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            avg_train = ep_loss / max(len(train_loader), 1)
            has_val = len(val_loader) > 0
            avg_val = _eval_sft_loss(model, val_loader, device, compute_dtype=compute_dtype) if has_val else avg_train
            selection_metric = avg_val if has_val else avg_train
            is_best = selection_metric < best_sft_loss
            if is_best:
                best_sft_loss = selection_metric
                best_sft_epoch = epoch + 1
                sft_best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(sft_best_dir)
                final_best_source = sft_best_dir

            sft_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(avg_train, 4),
                    "val_loss": round(avg_val, 4) if has_val else None,
                    "is_best": is_best,
                }
            )
            marker = " *best*" if is_best else ""
            val_str = f"{avg_val:.4f}" if has_val else "n/a"
            print(
                f"SFT Epoch {epoch + 1}/{epochs} | train_loss={avg_train:.4f} | val_loss={val_str}{marker}"
            )

        # Restore best SFT checkpoint before Stage 2 or final export
        if sft_best_dir.exists() and best_sft_epoch != epochs:
            _load_adapter_weights_into_model(model, sft_best_dir)
            print(f"Restored best SFT checkpoint from epoch {best_sft_epoch} (loss={best_sft_loss:.4f})")

        del optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stage 2: Reference-free SimPO preference optimization on hard negatives
    simpo_history: list[dict[str, Any]] = []
    best_simpo_loss = float("inf")
    best_simpo_epoch = 0

    if dpo_epochs > 0 and train_pref_triples:
        pref_train_ds = PreferenceDataset(train_pref_triples, tokenizer, max_len=max_length)
        pref_val_ds = PreferenceDataset(val_pref_triples, tokenizer, max_len=max_length)

        if len(pref_train_ds) > 0:
            pref_bs = max(1, simpo_batch_size)
            pref_train_loader = DataLoader(
                pref_train_ds,
                batch_size=pref_bs,
                shuffle=True,
                collate_fn=lambda b: _collate_pref(b, tokenizer.pad_token_id),
            )
            pref_val_loader = DataLoader(
                pref_val_ds,
                batch_size=pref_bs,
                shuffle=False,
                collate_fn=lambda b: _collate_pref(b, tokenizer.pad_token_id),
            )

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            pref_opt = _create_optimizer(
                trainable_params, lr=dpo_lr, weight_decay=0.01, use_paged_8bit=use_qlora
            )
            pref_steps = max(math.ceil(len(pref_train_loader) / grad_accum) * dpo_epochs, 1)
            pref_sched = get_cosine_schedule_with_warmup(
                pref_opt, num_warmup_steps=max(pref_steps // 10, 1), num_training_steps=pref_steps
            )

            print(
                f"=== Stage 2: SimPO ({dpo_epochs} epochs, {len(pref_train_ds)} train / {len(pref_val_ds)} val, "
                f"hard_neg={train_pref_stats['hard_negatives']}, src_fallback={train_pref_stats['source_fallback']}, "
                f"beta={simpo_beta}, gamma={simpo_gamma}, sft_anchor={sft_anchor_weight}) ==="
            )
            model.train()
            for epoch in range(dpo_epochs):
                ep_loss = 0.0
                ep_margin = 0.0
                ep_acc = 0.0
                pref_opt.zero_grad(set_to_none=True)
                for step, batch in enumerate(pref_train_loader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with _autocast_ctx(device, compute_dtype):
                        c_logp = _sequence_avg_logp(
                            model,
                            batch["chosen_input_ids"],
                            batch["chosen_attention_mask"],
                            batch["chosen_labels"],
                        )
                        r_logp = _sequence_avg_logp(
                            model,
                            batch["rejected_input_ids"],
                            batch["rejected_attention_mask"],
                            batch["rejected_labels"],
                        )
                        metrics = compute_simpo_loss(
                            c_logp,
                            r_logp,
                            beta=simpo_beta,
                            gamma=simpo_gamma,
                            sft_anchor_weight=sft_anchor_weight,
                        )
                    (metrics["loss"] / grad_accum).backward()
                    ep_loss += metrics["loss"].item()
                    ep_margin += metrics["margin"].item()
                    ep_acc += metrics["accuracy"].item()
                    if (step + 1) % grad_accum == 0 or (step + 1) == len(pref_train_loader):
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        pref_opt.step()
                        pref_sched.step()
                        pref_opt.zero_grad(set_to_none=True)

                n_steps = max(len(pref_train_loader), 1)
                avg_loss = ep_loss / n_steps
                avg_margin = ep_margin / n_steps
                avg_acc = ep_acc / n_steps
                has_pref_val = len(pref_val_loader) > 0
                if has_pref_val:
                    val_loss, val_margin, val_acc = _eval_simpo_loss(
                        model,
                        pref_val_loader,
                        device,
                        simpo_beta,
                        simpo_gamma,
                        sft_anchor_weight,
                        compute_dtype=compute_dtype,
                    )
                else:
                    val_loss, val_margin, val_acc = avg_loss, avg_margin, avg_acc

                selection_metric = val_loss if has_pref_val else avg_loss
                is_best = selection_metric < best_simpo_loss
                if is_best:
                    best_simpo_loss = selection_metric
                    best_simpo_epoch = epoch + 1
                    simpo_best_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(simpo_best_dir)
                    final_best_source = simpo_best_dir

                simpo_history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": round(avg_loss, 4),
                        "train_margin": round(avg_margin, 4),
                        "train_accuracy": round(avg_acc, 4),
                        "val_loss": round(val_loss, 4) if has_pref_val else None,
                        "val_margin": round(val_margin, 4) if has_pref_val else None,
                        "val_accuracy": round(val_acc, 4) if has_pref_val else None,
                        "is_best": is_best,
                    }
                )
                marker = " *best*" if is_best else ""
                val_str = f"{val_loss:.4f}" if has_pref_val else "n/a"
                print(
                    f"SimPO Epoch {epoch + 1}/{dpo_epochs} | train_loss={avg_loss:.4f} "
                    f"margin={avg_margin:+.4f} acc={avg_acc:.3f} | val_loss={val_str} "
                    f"val_margin={val_margin:+.4f}{marker}"
                )

            if simpo_best_dir.exists() and best_simpo_epoch != dpo_epochs:
                _load_adapter_weights_into_model(model, simpo_best_dir)
                print(
                    f"Restored best SimPO checkpoint from epoch {best_simpo_epoch} "
                    f"(loss={best_simpo_loss:.4f})"
                )

    elapsed = round(time.time() - t0, 1)
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else None
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Clean up intermediate stage checkpoint directories after saving best weights to output_dir
    for tmp_dir in (sft_best_dir, simpo_best_dir):
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    pushed_repo: str | None = None
    if push_to_hub:
        print(f"Pushing best adapter and tokenizer to Hugging Face Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub, private=hub_private)
        tokenizer.push_to_hub(push_to_hub, private=hub_private)
        pushed_repo = push_to_hub

    metrics_report: dict[str, Any] = {
        "base_model": base_model,
        "dataset_path": str(pairs_path),
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
        "use_4bit": use_qlora,
        "max_length": max_length,
        "seed": seed,
        "val_ratio": val_ratio,
        "identity_ratio": identity_ratio,
        "filter_stats": filter_stats,
        "split_stats": {
            "train_records": len(train_records),
            "val_records": len(val_records),
            "train_preference_triples": len(train_pref_triples),
            "val_preference_triples": len(val_pref_triples),
            "train_preference_sources": train_pref_stats,
            "val_preference_sources": val_pref_stats,
        },
        "hyperparameters": {
            "sft_epochs": epochs,
            "dpo_epochs": dpo_epochs,
            "lr": lr,
            "dpo_lr": dpo_lr,
            "simpo_beta": simpo_beta,
            "simpo_gamma": simpo_gamma,
            "sft_anchor_weight": sft_anchor_weight,
            "batch_size": batch_size,
            "simpo_batch_size": simpo_batch_size,
            "grad_accum": grad_accum,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
        "best_checkpoints": {
            "sft_best_epoch": best_sft_epoch if best_sft_epoch > 0 else None,
            "sft_best_loss": round(best_sft_loss, 4) if math.isfinite(best_sft_loss) else None,
            "simpo_best_epoch": best_simpo_epoch if best_simpo_epoch > 0 else None,
            "simpo_best_loss": round(best_simpo_loss, 4) if math.isfinite(best_simpo_loss) else None,
            "selected_stage": (
                "simpo"
                if final_best_source == simpo_best_dir
                else ("sft" if final_best_source == sft_best_dir else "initial")
            ),
        },
        "sft_history": sft_history,
        "simpo_history": simpo_history,
        "sft_train_loss_history": [h["train_loss"] for h in sft_history],
        "sft_val_loss_history": [h["val_loss"] for h in sft_history],
        "peak_vram_mb": peak_vram_mb,
        "train_seconds": elapsed,
        "pushed_to_hub": pushed_repo,
    }

    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    with open(output_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    return metrics_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m psycho_rev.train_lora",
        description=(
            "Train a QLoRA/LoRA SFT + SimPO adapter from an external distilled JSON/JSONL dataset "
            "or via Best-of-N self-distillation."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to external distilled JSON or JSONL dataset (skips self-distillation).",
    )
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=Path("results/distilled_train_pairs.json"),
        help="Path to save/load self-distilled training pairs JSON when --dataset-path is not set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/qwen25-3b-psycho-rev-lora"),
        help="Directory to save the best trained LoRA adapter and training_metrics.json.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Optional path to an existing LoRA adapter checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--teacher-model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Teacher model ID for Best-of-N self-distillation.",
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
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size for SFT.")
    parser.add_argument("--simpo-batch-size", type=int, default=1, help="Per-device batch size for SimPO.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length in tokens.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting and training.")
    parser.add_argument(
        "--identity-ratio",
        type=float,
        default=0.15,
        help="Maximum fraction of identity/no-op examples (target == source) in the dataset.",
    )
    parser.add_argument(
        "--no-builtin-identity",
        action="store_true",
        help="Do not supplement dataset with built-in IDENTITY_REGULARIZERS.",
    )
    parser.add_argument("--min-length-ratio", type=float, default=0.65, help="Minimum allowed target/source length ratio.")
    parser.add_argument("--max-length-ratio", type=float, default=1.85, help="Maximum allowed target/source length ratio.")
    parser.add_argument("--min-nli-entailment", type=float, default=0.70, help="Minimum NLI entailment when present.")
    parser.add_argument("--max-nli-contradiction", type=float, default=0.15, help="Maximum NLI contradiction when present.")
    parser.add_argument("--min-cosine-similarity", type=float, default=0.75, help="Minimum cosine similarity when present.")
    parser.add_argument("--min-acceptability", type=float, default=0.60, help="Minimum acceptability score when present.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA and train in bf16/fp16.")
    parser.add_argument("--nli-model", default=None, help="Optional NLI cross-encoder model override.")
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Optional Hugging Face Hub repo ID to push the best trained adapter to (off by default).",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create/push the Hugging Face Hub repository as private.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    effective_pairs_path = args.dataset_path if args.dataset_path is not None else args.pairs_path
    use_external_dataset = args.dataset_path is not None

    if (
        not use_external_dataset
        and not args.skip_distill
        and (args.resume or args.distill_only or not effective_pairs_path.exists())
    ):
        build_distillation_pairs(
            out_pairs_path=effective_pairs_path,
            teacher_model=args.teacher_model,
            num_samples=args.num_samples,
            max_samples=args.max_samples,
            resume=args.resume,
            nli_model=args.nli_model,
        )

    if args.distill_only:
        print(f"Distillation complete. Pairs saved to {effective_pairs_path}")
        return

    meta = train_lora_adapter(
        pairs_path=effective_pairs_path,
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
        simpo_batch_size=args.simpo_batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        use_4bit=not args.no_4bit,
        val_ratio=args.val_ratio,
        seed=args.seed,
        identity_ratio=args.identity_ratio,
        include_builtin_identity=not args.no_builtin_identity,
        min_length_ratio=args.min_length_ratio,
        max_length_ratio=args.max_length_ratio,
        min_nli_entailment=args.min_nli_entailment,
        max_nli_contradiction=args.max_nli_contradiction,
        min_cosine_similarity=args.min_cosine_similarity,
        min_acceptability=args.min_acceptability,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        resume_from_checkpoint=args.resume_from_checkpoint,
        push_to_hub=args.push_to_hub,
        hub_private=args.hub_private,
    )
    print("TRAINING COMPLETE:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
