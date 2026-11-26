"""CPU-only unit tests for src/psycho_rev/train_lora.py (no model downloads)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from psycho_rev.code_guard import protect_spans
from psycho_rev.neural import REWRITE_SYSTEM_PROMPT
from psycho_rev.train_lora import (
    PreferenceDataset,
    SFTDataset,
    _collate_pref,
    _collate_sft,
    _sequence_avg_logp,
    build_arg_parser,
    build_preference_triples,
    compute_length_ratio,
    compute_simpo_loss,
    encode_pair,
    filter_distilled_records,
    format_rewrite_messages,
    is_truncated_text,
    load_distilled_dataset,
    sequence_avg_logp_from_logits,
    split_train_val,
)


class _DummyChatTokenizer:
    """Minimal deterministic tokenizer implementing apply_chat_template for CPU tests."""

    pad_token_id = 0

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"<pad>": 0, "<bos>": 1, "<sep>": 2, "<gen>": 3}

    def _encode_text(self, text: str) -> list[int]:
        ids: list[int] = []
        for tok in text.split():
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
            ids.append(self._vocab[tok])
        return ids

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> list[int] | str:
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        rendered = " <sep> ".join(parts)
        if not tokenize:
            return rendered
        ids = [1] + self._encode_text(rendered)
        if add_generation_prompt:
            ids.append(3)
        return ids


class _TinyCausalLM(nn.Module):
    """Tiny synthetic causal LM returning logits over a small vocabulary."""

    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        h = self.emb(input_ids)
        logits = self.head(h)
        return SimpleNamespace(logits=logits)


# ---------------------------------------------------------------------------
# 1. JSON / JSONL dataset loading
# ---------------------------------------------------------------------------


def test_load_distilled_dataset_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "distilled.jsonl"
    rows = [
        {
            "id": "ex_1",
            "source": "The cache layer swallows `TimeoutError` in src/cache/store.py.",
            "target": "The cache layer ignores `TimeoutError` in src/cache/store.py.",
            "rejected": "The cache layer silently eats errors.",
            "domain": "technical",
            "nli_min_entailment": 0.91,
        },
        {
            "id": "ex_2",
            "source": "Keep this sentence clear and direct.",
            "target": "Keep this sentence clear and direct.",
            "domain": "prose",
        },
    ]
    path.write_text("\n\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")

    loaded = load_distilled_dataset(path)
    assert len(loaded) == 2
    assert loaded[0]["chosen"] == rows[0]["target"]
    assert loaded[0]["rejected"] == rows[0]["rejected"]
    assert loaded[0]["is_identity"] is False
    assert loaded[0]["domain"] == "technical"
    assert loaded[1]["is_identity"] is True
    assert loaded[1]["domain"] == "prose"


def test_load_distilled_dataset_json_list_and_wrapper(tmp_path: Path) -> None:
    list_path = tmp_path / "pairs_list.json"
    list_path.write_text(
        json.dumps([{"source": "Alpha beta.", "target": "Alpha and beta."}]),
        encoding="utf-8",
    )
    loaded_list = load_distilled_dataset(list_path)
    assert len(loaded_list) == 1
    assert loaded_list[0]["domain"] == "unknown"

    wrap_path = tmp_path / "pairs_wrap.json"
    wrap_path.write_text(
        json.dumps({"records": [{"source": "One two.", "chosen": "One and two."}]}),
        encoding="utf-8",
    )
    loaded_wrap = load_distilled_dataset(wrap_path)
    assert len(loaded_wrap) == 1
    assert loaded_wrap[0]["target"] == "One and two."


def test_load_distilled_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_distilled_dataset(tmp_path / "does_not_exist.jsonl")


# ---------------------------------------------------------------------------
# 2. Strict quality filtering & identity balancing
# ---------------------------------------------------------------------------


def test_filter_rejects_code_corruption_extreme_length_truncation_and_bad_metadata() -> None:
    records: list[dict[str, Any]] = [
        # Valid rewrite with preserved code span and URL
        {
            "source": "The worker calls `flush_cache()` at https://example.com/api and drops stale entries.",
            "target": "The worker invokes `flush_cache()` at https://example.com/api and removes stale entries.",
            "domain": "tech",
            "nli_min_entailment": 0.88,
            "nli_max_contradiction": 0.02,
            "cosine_similarity": 0.90,
            "acceptability_score": 0.85,
        },
        # Corrupted code span (`flush_cache()` removed)
        {
            "source": "The worker calls `flush_cache()` in store.py to clear memory.",
            "target": "The worker clears memory in store.py when needed.",
            "domain": "tech",
        },
        # Metadata reports code_preservation < 1.0 even if text has no detected spans
        {
            "source": "This sentence has plain words only.",
            "target": "This sentence uses plain words only.",
            "code_preservation": 0.5,
        },
        # Extreme short length ratio (< 0.65)
        {
            "source": "One two three four five six seven eight nine ten.",
            "target": "One two three.",
        },
        # Extreme long length ratio (> 1.85)
        {
            "source": "The server restarted quickly.",
            "target": "The server restarted very quickly after the administrator finished checking every single log file on disk.",
        },
        # Truncated output (unbalanced code fence)
        {
            "source": "Run the helper script to check the database.",
            "target": "Run the helper script:\n```python\ncheck()",
        },
        # Truncated output (explicit flag)
        {
            "source": "Check the configuration before deploying the service.",
            "target": "Review the configuration before deploying the service.",
            "truncated": True,
        },
        # Low NLI entailment
        {
            "source": "The database commits transactions synchronously to disk.",
            "target": "The database writes transactions synchronously to disk.",
            "nli_min_entailment": 0.40,
        },
        # High NLI contradiction
        {
            "source": "The job runs every hour on the primary node.",
            "target": "The job runs every hour on the backup node.",
            "nli_max_contradiction": 0.65,
        },
        # Low acceptability
        {
            "source": "The parser validates every incoming request header.",
            "target": "The parser check every incoming request header.",
            "acceptability_score": 0.30,
        },
        # Valid identity example (target == source)
        {
            "source": "The function returns an empty list when no files match.",
            "target": "The function returns an empty list when no files match.",
            "is_identity": True,
            "domain": "identity",
        },
    ]

    kept, stats = filter_distilled_records(
        records,
        identity_ratio=0.50,
        include_builtin_identity=False,
        seed=42,
    )

    assert stats["rejected_code_preservation"] == 2
    assert stats["rejected_length_ratio"] == 2
    assert stats["rejected_empty_or_truncated"] == 2
    assert stats["rejected_nli_entailment"] == 1
    assert stats["rejected_nli_contradiction"] == 1
    assert stats["rejected_acceptability"] == 1
    assert stats["kept_rewrite_records"] == 1
    assert stats["kept_identity_records"] == 1
    assert len(kept) == 2
    assert any(r["is_identity"] for r in kept)
    assert any(not r["is_identity"] for r in kept)


def test_filter_caps_identity_ratio_deterministically() -> None:
    rewrites = [
        {
            "source": f"The dense service pipeline handler {i} bottlenecks throughput.",
            "target": f"The service pipeline handler {i} slows down throughput.",
            "is_identity": False,
            "domain": "tech",
        }
        for i in range(85)
    ]
    identities = [
        {
            "source": f"Clear human sentence number {i} stays easy to read.",
            "target": f"Clear human sentence number {i} stays easy to read.",
            "is_identity": True,
            "domain": "identity",
        }
        for i in range(40)
    ]

    kept_a, stats_a = filter_distilled_records(
        rewrites + identities,
        identity_ratio=0.15,
        include_builtin_identity=False,
        seed=42,
    )
    kept_b, _ = filter_distilled_records(
        rewrites + identities,
        identity_ratio=0.15,
        include_builtin_identity=False,
        seed=42,
    )

    assert kept_a == kept_b
    # floor(85 * 0.15 / 0.85) == 15 -> 15 / (85 + 15) == 0.15
    assert stats_a["kept_rewrite_records"] == 85
    assert stats_a["kept_identity_records"] == 15
    assert stats_a["identity_capped"] == 25
    assert stats_a["effective_identity_ratio"] == pytest.approx(0.15, abs=1e-4)

    kept_zero_id, stats_zero = filter_distilled_records(
        rewrites + identities,
        identity_ratio=0.0,
        include_builtin_identity=False,
        seed=42,
    )
    assert stats_zero["kept_identity_records"] == 0
    assert len(kept_zero_id) == 85


def test_truncation_and_length_ratio_helpers() -> None:
    assert compute_length_ratio("one two three four", "one two three") == pytest.approx(0.75)
    assert is_truncated_text("Hello world.", "") is True
    assert is_truncated_text("Hello world.", "Hello world...") is True
    assert is_truncated_text("Wait for it...", "Wait for it...") is False
    assert is_truncated_text("The server handles incoming requests cleanly.", "The server handles incoming requests cleanly,") is True
    assert is_truncated_text("The server handles incoming requests cleanly.", "The server handles incoming requests well.") is False


def test_filter_checks_code_preservation_rate_metadata_key() -> None:
    records = [
        {
            "source": "This sentence has plain words only.",
            "target": "This sentence uses plain words only.",
            "code_preservation_rate": 0.5,
        },
        {
            "source": "Another plain sentence stays intact.",
            "target": "Another plain sentence remains intact.",
            "code_preservation_rate": 1.0,
        },
    ]
    kept, stats = filter_distilled_records(records, include_builtin_identity=False)
    assert stats["rejected_code_preservation"] == 1
    assert len(kept) == 1
    assert kept[0]["source"] == "Another plain sentence stays intact."


# ---------------------------------------------------------------------------
# 3. Deterministic train/val split & hard-negative preference triples
# ---------------------------------------------------------------------------


def test_split_train_val_is_deterministic_disjoint_and_stratified() -> None:
    records = [
        {"id": f"rw_{i}", "source": f"s{i}", "target": f"t{i}", "is_identity": False, "domain": "tech"}
        for i in range(40)
    ] + [
        {"id": f"id_{i}", "source": f"i{i}", "target": f"i{i}", "is_identity": True, "domain": "identity"}
        for i in range(10)
    ]

    train1, val1 = split_train_val(records, val_ratio=0.2, seed=42)
    train2, val2 = split_train_val(records, val_ratio=0.2, seed=42)
    train3, val3 = split_train_val(records, val_ratio=0.2, seed=99)

    assert [r["id"] for r in train1] == [r["id"] for r in train2]
    assert [r["id"] for r in val1] == [r["id"] for r in val2]
    assert [r["id"] for r in val1] != [r["id"] for r in val3]

    train_ids = {r["id"] for r in train1}
    val_ids = {r["id"] for r in val1}
    assert train_ids.isdisjoint(val_ids)
    assert len(train1) == 40
    assert len(val1) == 10
    # Stratification keeps identity examples in both train and val
    assert any(r["is_identity"] for r in val1)
    assert any(not r["is_identity"] for r in val1)


def test_build_preference_triples_uses_hard_negatives_with_source_fallback() -> None:
    records = [
        {
            "source": "Dense source A.",
            "chosen": "Clear rewrite A.",
            "rejected": "Clunky over-edited A.",
        },
        {
            "source": "Dense source B.",
            "chosen": "Clear rewrite B.",
        },
        {
            "source": "Already clear C.",
            "chosen": "Already clear C.",
            "is_identity": True,
        },
        {
            "source": "Already clear D.",
            "chosen": "Already clear D.",
            "rejected": "Unnecessarily bloated rewrite of D.",
            "is_identity": True,
        },
    ]

    triples, counts = build_preference_triples(records, fallback_to_source=True)
    assert counts == {"hard_negatives": 2, "source_fallback": 1, "skipped": 1}
    assert triples[0] == ("Dense source A.", "Clear rewrite A.", "Clunky over-edited A.")
    assert triples[1] == ("Dense source B.", "Clear rewrite B.", "Dense source B.")
    assert triples[2] == ("Already clear D.", "Already clear D.", "Unnecessarily bloated rewrite of D.")


# ---------------------------------------------------------------------------
# 4. Chat-template formatting, encoding & collation
# ---------------------------------------------------------------------------


def test_format_rewrite_messages_uses_canonical_system_prompt_and_code_guard() -> None:
    src = "Check src/cache/store.py and store.py using `is_file()` at https://example.com/v1."
    tgt = "Inspect src/cache/store.py and store.py with `is_file()` at https://example.com/v1."

    msgs = format_rewrite_messages(src, tgt)
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == REWRITE_SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"].startswith("Rewrite this passage clearly:\n\n")
    assert "src/cache/store.py" not in msgs[1]["content"]
    assert "`is_file()`" not in msgs[1]["content"]
    assert "https://example.com/v1" not in msgs[1]["content"]
    # Assistant target must use the exact same placeholders (no substring collision on store.py)
    assert "src/cache/store.py" not in msgs[2]["content"]
    assert "store.py" not in msgs[2]["content"]
    assert "CODEITEM0" in msgs[1]["content"] and "CODEITEM0" in msgs[2]["content"]
    assert "CODEITEM1" in msgs[1]["content"] and "CODEITEM1" in msgs[2]["content"]


def test_format_rewrite_messages_preserves_duplicate_code_spans_and_identity() -> None:
    src = "Call `foo()` first, then call `foo()` again."
    tgt = "First invoke `foo()`, then invoke `foo()` again."
    guarded = protect_spans(src)

    msgs = format_rewrite_messages(src, tgt)
    assert "CODEITEM0" in msgs[1]["content"] and "CODEITEM1" in msgs[1]["content"]
    assert "CODEITEM0" in msgs[2]["content"] and "CODEITEM1" in msgs[2]["content"]
    assert msgs[2]["content"].count("CODEITEM0") == 1
    assert msgs[2]["content"].count("CODEITEM1") == 1
    assert guarded.restore(msgs[2]["content"]) == tgt

    # Identity pair maps directly to g_src.masked
    id_msgs = format_rewrite_messages(src, src)
    assert id_msgs[2]["content"] == guarded.masked
    assert "CODEITEM0" in id_msgs[2]["content"] and "CODEITEM1" in id_msgs[2]["content"]


class _EOSDummyChatTokenizer(_DummyChatTokenizer):
    """Dummy chat tokenizer that appends an explicit EOS token to complete assistant turns."""

    eos_token_id = 4

    def __init__(self) -> None:
        super().__init__()
        self._vocab["<eos>"] = 4

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> list[int] | str:
        out = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if not tokenize:
            return out if add_generation_prompt else f"{out} <eos>"
        assert isinstance(out, list)
        if not add_generation_prompt:
            return out + [self.eos_token_id]
        return out


def test_encode_pair_never_emits_truncated_eos_and_splits_multipara() -> None:
    tok = _EOSDummyChatTokenizer()
    src = "The service processes incoming webhook events sequentially."
    tgt = "The service handles incoming webhook events one at a time."

    full_enc = encode_pair(src, tgt, tok, max_len=512)
    assert full_enc is not None
    full_len = len(full_enc["input_ids"])
    assert full_enc["input_ids"][-1].item() == tok.eos_token_id
    assert full_enc["labels"][-1].item() == tok.eos_token_id

    # Even 1 token shorter than full_len would drop EOS, so encode_pair must return None
    assert encode_pair(src, tgt, tok, max_len=full_len - 1) is None

    sft_dropped = SFTDataset([(src, tgt)], tok, max_len=full_len - 1)
    assert len(sft_dropped) == 0
    assert sft_dropped.dropped_truncated == 1

    pref_dropped = PreferenceDataset([(src, tgt, "Clunky rewrite here.")], tok, max_len=full_len - 1)
    assert len(pref_dropped) == 0
    assert pref_dropped.dropped_truncated == 1

    # Multi-paragraph aligned pair exceeding max_len as a whole doc splits into paragraph pairs with intact EOS
    p1_src = "First paragraph describes how the worker drains queued jobs."
    p1_tgt = "First paragraph explains how the worker handles queued jobs."
    p1_rej = "First paragraph bloatedly describes queued job draining."
    p2_src = "Second paragraph explains why retries use exponential backoff."
    p2_tgt = "Second paragraph shows why retries use exponential backoff."
    p2_rej = "Second paragraph awkwardly discusses retry timing."

    whole_src = f"{p1_src}\n\n{p2_src}"
    whole_tgt = f"{p1_tgt}\n\n{p2_tgt}"
    whole_rej = f"{p1_rej}\n\n{p2_rej}"

    p1_len = len(encode_pair(p1_src, p1_tgt, tok, max_len=512)["input_ids"])
    p2_len = len(encode_pair(p2_src, p2_tgt, tok, max_len=512)["input_ids"])
    whole_len = len(encode_pair(whole_src, whole_tgt, tok, max_len=512)["input_ids"])
    split_cap = max(p1_len, p2_len) + 2
    assert split_cap < whole_len

    sft_split = SFTDataset([(whole_src, whole_tgt)], tok, max_len=split_cap)
    assert len(sft_split) == 2
    assert sft_split.split_multipara_docs == 1
    assert sft_split.dropped_truncated == 0
    for item in sft_split.items:
        assert item["input_ids"][-1].item() == tok.eos_token_id
        assert item["labels"][-1].item() == tok.eos_token_id

    pref_split = PreferenceDataset([(whole_src, whole_tgt, whole_rej)], tok, max_len=split_cap)
    assert len(pref_split) == 2
    assert pref_split.split_multipara_docs == 1
    assert pref_split.dropped_truncated == 0
    for item in pref_split.items:
        assert item["chosen_input_ids"][-1].item() == tok.eos_token_id
        assert item["rejected_input_ids"][-1].item() == tok.eos_token_id


def test_encode_pair_masks_prompt_and_collates_cleanly() -> None:
    tok = _DummyChatTokenizer()
    enc = encode_pair("The cache swallows errors.", "The cache ignores errors.", tok, max_len=512)
    input_ids = enc["input_ids"]
    labels = enc["labels"]

    assert input_ids.shape == labels.shape
    prompt_ids = tok.apply_chat_template(
        format_rewrite_messages("The cache swallows errors."),
        tokenize=True,
        add_generation_prompt=True,
    )
    assert (labels[: len(prompt_ids)] == -100).all()
    assert (labels[len(prompt_ids) :] == input_ids[len(prompt_ids) :]).all()

    # Truncation below prompt length produces zero response tokens and is dropped by SFTDataset
    ds = SFTDataset(
        [
            ("Short source.", "Short target."),
            ("Another source.", "Another target."),
        ],
        tok,
        max_len=10,
    )
    assert len(ds) == 0
    assert ds.dropped_truncated == 2

    # Normal collation pads input_ids with pad_id, attention_mask with 0, labels with -100
    sft_ds = SFTDataset(
        [
            ("One two.", "One and two."),
            ("Three four five six.", "Three, four, five, and six."),
        ],
        tok,
        max_len=512,
    )
    batch = _collate_sft([sft_ds[0], sft_ds[1]], pad_id=tok.pad_token_id)
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape

    pref_ds = PreferenceDataset(
        [("One two.", "One and two.", "One two bad.")],
        tok,
        max_len=512,
    )
    pref_batch = _collate_pref([pref_ds[0]], pad_id=tok.pad_token_id)
    assert set(pref_batch.keys()) == {
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_labels",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_labels",
    }


# ---------------------------------------------------------------------------
# 5. SimPO loss math & gradient flow on tiny synthetic tensors
# ---------------------------------------------------------------------------


def test_sequence_avg_logp_matches_reference_log_softmax_gather() -> None:
    torch.manual_seed(7)
    batch_size, seq_len, vocab = 3, 8, 17
    logits = torch.randn(batch_size, seq_len, vocab, dtype=torch.float32)
    labels = torch.tensor(
        [
            [-100, -100, -100, 4, 7, 2, -100, -100],
            [-100, -100, 1, 5, 9, 12, 3, 6],
            [-100, -100, -100, -100, -100, 8, 10, -100],
        ],
        dtype=torch.long,
    )

    got = sequence_avg_logp_from_logits(logits, labels)

    # Reference implementation via explicit log_softmax + gather
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.clamp(min=0)
    token_logp = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    expected = (token_logp * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)


def test_sequence_avg_logp_is_length_normalized() -> None:
    vocab = 5
    # Construct identical per-position logits so every response token has the same log-prob
    base_row = torch.tensor([0.0, 2.0, -1.0, 0.5, 0.0])
    logits_short = base_row.view(1, 1, vocab).expand(1, 4, vocab).clone()
    labels_short = torch.tensor([[-100, -100, 1, 1]], dtype=torch.long)

    logits_long = base_row.view(1, 1, vocab).expand(1, 8, vocab).clone()
    labels_long = torch.tensor([[-100, -100, 1, 1, 1, 1, 1, 1]], dtype=torch.long)

    lp_short = sequence_avg_logp_from_logits(logits_short, labels_short)
    lp_long = sequence_avg_logp_from_logits(logits_long, labels_long)
    assert torch.allclose(lp_short, lp_long, atol=1e-6)


def test_compute_simpo_loss_exact_formula_and_monotonicity() -> None:
    c_logp = torch.tensor([-0.4, -0.6], dtype=torch.float32)
    r_logp = torch.tensor([-1.2, -0.5], dtype=torch.float32)
    beta, gamma, anchor = 2.0, 0.5, 0.1

    out = compute_simpo_loss(c_logp, r_logp, beta=beta, gamma=gamma, sft_anchor_weight=anchor)

    expected_margins = c_logp - r_logp
    expected_simpo = -F.logsigmoid(beta * expected_margins - gamma).mean()
    expected_anchor = -c_logp.mean()
    expected_total = expected_simpo + anchor * expected_anchor

    assert math.isclose(out["simpo_loss"].item(), expected_simpo.item(), rel_tol=1e-6)
    assert math.isclose(out["sft_anchor"].item(), expected_anchor.item(), rel_tol=1e-6)
    assert math.isclose(out["loss"].item(), expected_total.item(), rel_tol=1e-6)
    assert math.isclose(out["accuracy"].item(), 0.5, rel_tol=1e-6)

    # Higher chosen logp should strictly decrease SimPO loss
    better_out = compute_simpo_loss(c_logp + 0.5, r_logp, beta=beta, gamma=gamma, sft_anchor_weight=anchor)
    assert better_out["loss"].item() < out["loss"].item()


def test_simpo_backward_step_on_tiny_synthetic_model() -> None:
    torch.manual_seed(0)
    model = _TinyCausalLM(vocab_size=24, hidden_dim=8)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    c_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    c_mask = torch.ones_like(c_ids)
    c_labels = torch.tensor([[-100, -100, 3, 4, 5]], dtype=torch.long)

    r_ids = torch.tensor([[1, 2, 6, 7, 8]], dtype=torch.long)
    r_mask = torch.ones_like(r_ids)
    r_labels = torch.tensor([[-100, -100, 6, 7, 8]], dtype=torch.long)

    c_logp_before = _sequence_avg_logp(model, c_ids, c_mask, c_labels)
    r_logp_before = _sequence_avg_logp(model, r_ids, r_mask, r_labels)
    loss_before = compute_simpo_loss(c_logp_before, r_logp_before, beta=2.0, gamma=0.5, sft_anchor_weight=0.1)

    opt.zero_grad()
    loss_before["loss"].backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
    opt.step()

    with torch.no_grad():
        c_logp_after = _sequence_avg_logp(model, c_ids, c_mask, c_labels)
        r_logp_after = _sequence_avg_logp(model, r_ids, r_mask, r_labels)
        loss_after = compute_simpo_loss(c_logp_after, r_logp_after, beta=2.0, gamma=0.5, sft_anchor_weight=0.1)

    assert loss_after["loss"].item() < loss_before["loss"].item()


# ---------------------------------------------------------------------------
# 6. CLI argument defaults & flags
# ---------------------------------------------------------------------------


def test_cli_arg_parser_flags_and_defaults() -> None:
    parser = build_arg_parser()
    defaults = parser.parse_args([])
    assert defaults.dataset_path is None
    assert defaults.val_ratio == pytest.approx(0.1)
    assert defaults.seed == 42
    assert defaults.identity_ratio == pytest.approx(0.15)
    assert defaults.max_length == 1024
    assert defaults.push_to_hub is None
    assert defaults.resume_from_checkpoint is None

    parsed = parser.parse_args(
        [
            "--dataset-path",
            "data/distilled.jsonl",
            "--max-length",
            "1024",
            "--identity-ratio",
            "0.12",
            "--val-ratio",
            "0.15",
            "--seed",
            "7",
            "--push-to-hub",
            "org/psycho-rev-lora",
        ]
    )
    assert parsed.dataset_path == Path("data/distilled.jsonl")
    assert parsed.max_length == 1024
    assert parsed.identity_ratio == pytest.approx(0.12)
    assert parsed.val_ratio == pytest.approx(0.15)
    assert parsed.seed == 7
    assert parsed.push_to_hub == "org/psycho-rev-lora"
