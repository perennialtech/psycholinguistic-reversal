"""CPU-only unit tests for semantic evaluation helpers and benchmark dataset integrity."""

from __future__ import annotations

import contextlib
import inspect
import math
import re
import sys
from collections import Counter
from types import ModuleType, SimpleNamespace

import numpy as np

from psycho_rev.benchmark_data import (
    HUMAN_CONTROL,
    HUMAN_CONTROL_MULTIPARA,
    IDENTITY_REGULARIZERS,
    MULTIPARA_BENCHMARK,
    TEST_BENCHMARK,
    TRAIN_DENSE_CORPUS,
)
from psycho_rev.semantics import (
    SemanticEvaluator,
    SemanticReport,
    align_text_chunks,
    resolve_nli_label_indices,
    softmax_nli_probs,
    strip_fenced_code,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def test_resolve_nli_label_indices_standard_layouts():
    deberta_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
    assert resolve_nli_label_indices(deberta_map) == (1, 0)

    mnli_alt_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    assert resolve_nli_label_indices(mnli_alt_map) == (0, 2)

    string_keys_map = {"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"}
    assert resolve_nli_label_indices(string_keys_map) == (2, 0)


def test_resolve_nli_label_indices_fallbacks():
    assert resolve_nli_label_indices(None) == (1, 0)
    assert resolve_nli_label_indices({}) == (1, 0)
    assert resolve_nli_label_indices({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}) == (1, 0)


def test_softmax_nli_probs_extraction_and_stability():
    # Layout: {0: contradiction, 1: entailment, 2: neutral}
    logits = np.array([1.0, 3.0, 0.0], dtype=np.float64)
    entail, contra = softmax_nli_probs(logits, entail_idx=1, contra_idx=0)
    denom = math.exp(1.0) + math.exp(3.0) + math.exp(0.0)
    assert math.isclose(entail, math.exp(3.0) / denom, rel_tol=1e-6)
    assert math.isclose(contra, math.exp(1.0) / denom, rel_tol=1e-6)

    # Alternate layout: {0: entailment, 1: neutral, 2: contradiction}
    entail_alt, contra_alt = softmax_nli_probs(logits, entail_idx=0, contra_idx=2)
    assert math.isclose(entail_alt, math.exp(1.0) / denom, rel_tol=1e-6)
    assert math.isclose(contra_alt, math.exp(0.0) / denom, rel_tol=1e-6)

    # Large shifted logits must not overflow
    shifted_entail, shifted_contra = softmax_nli_probs(logits + 1000.0, entail_idx=1, contra_idx=0)
    assert math.isclose(shifted_entail, entail, rel_tol=1e-6)
    assert math.isclose(shifted_contra, contra, rel_tol=1e-6)


def test_default_nli_model_and_report_fields():
    sig = inspect.signature(SemanticEvaluator.__init__)
    assert sig.parameters["nli_model"].default == "cross-encoder/nli-deberta-v3-base"

    report = SemanticReport(
        cosine_similarity=0.91,
        nli_forward_entailment=0.88,
        nli_backward_entailment=0.84,
        nli_min_entailment=0.84,
        gpt2_perplexity=24.5,
        code_preservation=1.0,
        length_ratio=0.95,
    )
    assert report.nli_max_contradiction == 0.0
    as_dict = report.to_dict()
    for field in (
        "cosine_similarity",
        "nli_forward_entailment",
        "nli_backward_entailment",
        "nli_min_entailment",
        "nli_max_contradiction",
        "gpt2_perplexity",
        "code_preservation",
        "length_ratio",
    ):
        assert field in as_dict


def test_align_text_chunks_short_and_long():
    short_a = "This is a short sentence."
    short_b = "This is another short sentence."
    assert align_text_chunks(short_a, short_b) == [(short_a, short_b)]

    para1 = " ".join(f"word{i}" for i in range(120))
    para2 = " ".join(f"term{i}" for i in range(120))
    long_orig = f"{para1}\n\n{para2}"
    long_rew = f"{para1}\n\n{para2}"
    aligned = align_text_chunks(long_orig, long_rew)
    assert len(aligned) == 2
    assert aligned[0] == (para1, para1)
    assert aligned[1] == (para2, para2)

    # Single long paragraph > 200 words splits into ~150-word chunks
    single_long = " ".join(f"token{i}." for i in range(320))
    aligned_single = align_text_chunks(single_long, single_long)
    assert len(aligned_single) >= 3


def test_strip_fenced_code_and_mocked_evaluator_pipeline(monkeypatch):
    text_with_code = "Intro text.\n\n```python\nx = 1 + 2\n```\n\nOutro text."
    stripped = strip_fenced_code(text_with_code)
    assert "x = 1 + 2" not in stripped
    assert "Intro text." in stripped and "Outro text." in stripped

    # Provide a lightweight torch stub only when torch is absent in minimal CPU test envs
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False

    class _FakeTensor:
        def __init__(self, arr: np.ndarray):
            self._arr = np.asarray(arr)

        @property
        def shape(self):
            return self._arr.shape

        def to(self, _device):
            return self

        def clone(self):
            return _FakeTensor(self._arr.copy())

        def __getitem__(self, key):
            return _FakeTensor(self._arr[key])

        def __setitem__(self, key, value):
            self._arr[key] = value

    if not has_torch:
        fake_torch = ModuleType("torch")
        fake_torch.inference_mode = contextlib.nullcontext
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    ev = SemanticEvaluator.__new__(SemanticEvaluator)
    ev.device = "cpu"
    ev._entail_idx, ev._contra_idx = resolve_nli_label_indices(
        {0: "entailment", 1: "neutral", 2: "contradiction"}
    )

    class _MockEmbedder:
        def encode(self, texts, normalize_embeddings=True, batch_size=32):
            return np.ones((len(texts), 4), dtype=np.float32) * 0.5

    class _MockNLI:
        def predict(self, pairs, batch_size=32):
            out = []
            for idx, _ in enumerate(pairs):
                if idx == 2:
                    out.append([0.5, 0.5, 2.5])  # contradiction-heavy on chunk 2 fwd
                else:
                    out.append([3.0, 0.5, -1.0])  # entailment-heavy
            return np.array(out, dtype=np.float64)

    class _MockTokenizer:
        def __call__(self, text, return_tensors="pt", truncation=False):
            n_tokens = max(len(text.split()), 1)
            if has_torch:
                import torch
                return {"input_ids": torch.arange(n_tokens, dtype=torch.long).unsqueeze(0)}
            return {"input_ids": _FakeTensor(np.arange(n_tokens, dtype=np.int64)[None, :])}

    class _MockLoss:
        def __init__(self, val: float):
            self._val = val

        def item(self) -> float:
            return self._val

    class _MockLM:
        def __init__(self):
            self.calls = 0

        def __call__(self, input_ids, labels=None):
            self.calls += 1
            return SimpleNamespace(loss=_MockLoss(2.0))

    ev.embedder = _MockEmbedder()
    ev.nli = _MockNLI()
    ev.lm_tok = _MockTokenizer()
    ev.lm = _MockLM()

    # Long 2-paragraph document (>200 words) triggers chunked NLI & strided PPL (>512 tokens)
    p1 = " ".join(f"alpha{i}" for i in range(300))
    p2 = " ".join(f"beta{i}" for i in range(300))
    doc = f"{p1}\n\n{p2}"

    reports = ev.evaluate_batch([doc], [doc])
    assert len(reports) == 1
    rep = reports[0]
    assert math.isclose(rep.cosine_similarity, 1.0, abs_tol=1e-4)
    assert rep.nli_max_contradiction > 0.7  # conservative max across chunks
    assert 0.0 < rep.nli_min_entailment < 1.0
    assert math.isclose(rep.gpt2_perplexity, round(math.exp(2.0), 2), abs_tol=1e-2)
    assert ev.lm.calls >= 2  # strided windowing invoked for 600-token sequence

    ev.close()
    assert ev.embedder is None and ev.nli is None and ev.lm is None and ev.lm_tok is None


def test_dataset_counts_structure_and_strict_disjointness():
    assert len(TEST_BENCHMARK) == 30
    assert len(HUMAN_CONTROL) == 10
    assert len(MULTIPARA_BENCHMARK) == 18
    assert len(HUMAN_CONTROL_MULTIPARA) == 6
    assert len(TRAIN_DENSE_CORPUS) >= 200
    assert len(IDENTITY_REGULARIZERS) >= 30

    # Verify MULTIPARA_BENCHMARK domain balance and length/paragraph bounds
    mp_domains = Counter(s.domain for s in MULTIPARA_BENCHMARK)
    assert mp_domains == {"chat": 6, "coding": 6, "essay": 6}
    for sample in MULTIPARA_BENCHMARK:
        words = len(sample.text.split())
        paras = [p for p in re.split(r"\n\s*\n+", sample.text.strip()) if p.strip()]
        assert 70 <= words <= 160, f"{sample.sample_id} has {words} words"
        assert 2 <= len(paras) <= 3, f"{sample.sample_id} has {len(paras)} paragraphs"
        if sample.domain == "coding":
            assert "```" in sample.text, f"{sample.sample_id} missing fenced code block"
            assert "\n- " in sample.text, f"{sample.sample_id} missing bullet list"

    for sample in HUMAN_CONTROL_MULTIPARA:
        words = len(sample.text.split())
        paras = [p for p in re.split(r"\n\s*\n+", sample.text.strip()) if p.strip()]
        assert 70 <= words <= 140, f"{sample.sample_id} has {words} words"
        assert 2 <= len(paras) <= 3, f"{sample.sample_id} has {len(paras)} paragraphs"

    # Strict disjointness: zero exact or normalized overlap between any train and test split
    test_splits = [
        ("TEST_BENCHMARK", [s.text for s in TEST_BENCHMARK]),
        ("HUMAN_CONTROL", [s.text for s in HUMAN_CONTROL]),
        ("MULTIPARA_BENCHMARK", [s.text for s in MULTIPARA_BENCHMARK]),
        ("HUMAN_CONTROL_MULTIPARA", [s.text for s in HUMAN_CONTROL_MULTIPARA]),
    ]
    train_splits = [
        ("TRAIN_DENSE_CORPUS", list(TRAIN_DENSE_CORPUS)),
        ("IDENTITY_REGULARIZERS", list(IDENTITY_REGULARIZERS)),
    ]

    # No duplicates within any split
    for name, texts in test_splits + train_splits:
        normed = [_normalize(t) for t in texts]
        assert len(set(normed)) == len(normed), f"Duplicate passages found in {name}"

    all_test_exact = {t for _, texts in test_splits for t in texts}
    all_test_norm = {_normalize(t) for t in all_test_exact}

    all_train_exact = {t for _, texts in train_splits for t in texts}
    all_train_norm = {_normalize(t) for t in all_train_exact}

    assert all_test_exact.isdisjoint(all_train_exact), "Exact overlap between train and test splits"
    assert all_test_norm.isdisjoint(all_train_norm), "Normalized overlap between train and test splits"
