"""CPU-only unit tests for semantic evaluation, CoLA acceptability, blind pairwise judge, and datasets."""

from __future__ import annotations

import contextlib
import inspect
import json
import math
import re
import sys
from collections import Counter
from types import ModuleType, SimpleNamespace

import httpx
import numpy as np

from psycho_rev.benchmark_data import (
    HUMAN_CONTROL,
    HUMAN_CONTROL_MULTIPARA,
    IDENTITY_REGULARIZERS,
    MULTIPARA_BENCHMARK,
    TEST_BENCHMARK,
    TRAIN_DENSE_CORPUS,
)
from psycho_rev.judge import (
    BlindPairwiseJudge,
    JudgeResult,
    PairwiseSummary,
    parse_judge_response,
    reconcile_swapped_verdicts,
    summarize_pairwise,
)
from psycho_rev.semantics import (
    DEFAULT_COLA_MODEL,
    DEFAULT_EVAL_NLI_MODEL,
    DEFAULT_HELDOUT_FLUENCY_MODEL,
    DEFAULT_RERANK_NLI_MODEL,
    AcceptabilityScorer,
    HeldOutSemanticEvaluator,
    SemanticEvaluator,
    SemanticReport,
    align_text_chunks,
    resolve_cola_label_index,
    resolve_nli_label_indices,
    softmax_nli_probs,
    split_prose_sentences,
    strip_fenced_code,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _ensure_torch_stub(monkeypatch) -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        fake_torch = ModuleType("torch")
        fake_torch.inference_mode = contextlib.nullcontext
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        return False


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


def test_resolve_cola_label_index():
    assert resolve_cola_label_index({0: "LABEL_0", 1: "LABEL_1"}) == 1
    assert resolve_cola_label_index({0: "unacceptable", 1: "acceptable"}) == 1
    assert resolve_cola_label_index({0: "acceptable", 1: "unacceptable"}) == 0
    assert resolve_cola_label_index({"0": "UNACCEPTABLE", "1": "LABEL_1"}) == 1
    assert resolve_cola_label_index(None) == 1


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
    assert sig.parameters["role"].default == "eval"
    assert sig.parameters["enable_acceptability"].default is True
    assert sig.parameters["acceptability_model"].default == DEFAULT_COLA_MODEL

    heldout_sig = inspect.signature(HeldOutSemanticEvaluator.__init__)
    assert heldout_sig.parameters["nli_model"].default == DEFAULT_EVAL_NLI_MODEL
    assert heldout_sig.parameters["lm_model"].default == DEFAULT_HELDOUT_FLUENCY_MODEL
    assert heldout_sig.parameters["enable_acceptability"].default is True
    assert DEFAULT_RERANK_NLI_MODEL == "cross-encoder/nli-deberta-v3-small"

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
    assert report.acceptability_score == 1.0
    assert report.acceptability_mean == 1.0
    assert report.acceptability_min == 1.0
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
        "acceptability_score",
        "acceptability_mean",
        "acceptability_min",
    ):
        assert field in as_dict


def test_split_prose_sentences_and_acceptability_scorer_catches_mangled_sentence(monkeypatch):
    _ensure_torch_stub(monkeypatch)

    passage = (
        "The team shipped the release on time. "
        "The quarterly leadership for roadmap alignment deck product circulated yesterday.\n\n"
        "```python\nx = 1 + 2\n```\n\n"
        "- All regression checks passed."
    )
    sents = split_prose_sentences(passage)
    assert len(sents) == 3
    assert all("x = 1 + 2" not in s for s in sents)

    scorer = AcceptabilityScorer(enabled=True, device="cpu")
    scorer._acceptable_idx = 1

    class _MockColaTok:
        def __call__(self, batch, padding=True, truncation=True, max_length=256, return_tensors="pt"):
            return {"sentences": batch}

    class _MockColaModel:
        def __call__(self, sentences):
            rows = []
            for s in sentences:
                if "roadmap alignment deck product" in s:
                    rows.append([3.5, -2.5])  # strongly unacceptable
                else:
                    rows.append([-2.5, 3.5])  # strongly acceptable
            return SimpleNamespace(logits=np.asarray(rows, dtype=np.float64))

    scorer._tokenizer = _MockColaTok()
    scorer._model = _MockColaModel()

    acc_mean, acc_min = scorer.score_text(passage)
    assert acc_min < 0.05
    assert 0.5 < acc_mean < 0.8
    assert scorer.is_acceptable(passage, min_threshold=0.5) is False
    assert scorer.is_acceptable("The team shipped the release on time.", min_threshold=0.5) is True

    # Batch scoring matches individual calls
    batch_res = scorer.score_texts(["The team shipped the release on time.", passage])
    assert batch_res[0][1] > 0.95
    assert batch_res[1] == (acc_mean, acc_min)

    scorer.close()
    assert not scorer.is_loaded


def test_acceptability_scorer_disabled_and_fallback():
    disabled = AcceptabilityScorer(enabled=False)
    assert disabled.score_text("Broken grammar word salad here.") == (1.0, 1.0)
    assert disabled.is_acceptable("Broken grammar word salad here.") is True

    broken = AcceptabilityScorer(model_name="nonexistent/local-only-cola", enabled=True)
    broken._load_failed = True
    assert broken.score_text("Any sentence here.") == (1.0, 1.0)


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

    has_torch = _ensure_torch_stub(monkeypatch)

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

    ev = SemanticEvaluator.__new__(SemanticEvaluator)
    ev.device = "cpu"
    ev.enable_acceptability = True
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

    class _MockAcceptability:
        def score_texts(self, texts):
            return [(0.82, 0.41) for _ in texts]

        def close(self):
            pass

    ev.embedder = _MockEmbedder()
    ev.nli = _MockNLI()
    ev.lm_tok = _MockTokenizer()
    ev.lm = _MockLM()
    ev.acceptability_scorer = _MockAcceptability()

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
    assert rep.acceptability_mean == 0.82
    assert rep.acceptability_min == 0.41
    assert rep.acceptability_score == 0.41
    assert ev.lm.calls >= 2  # strided windowing invoked for 600-token sequence

    ev.close()
    assert ev.embedder is None and ev.nli is None and ev.lm is None and ev.lm_tok is None
    assert ev.acceptability_scorer is None


def test_blind_judge_parse_and_reconcile_swapped_verdicts():
    raw_fenced = """```json
    {
      "candidate_1": {"clarity": 5, "naturalness": 4, "meaning_preservation": 5},
      "candidate_2": {"clarity": 2, "naturalness": 3, "meaning_preservation": 4},
      "winner": "1",
      "rationale": "Candidate 1 is much clearer."
    }
    ```"""
    parsed_ab = parse_judge_response(raw_fenced)
    assert parsed_ab["winner"] == "1"
    assert parsed_ab["candidate_1"]["clarity"] == 5.0

    # Consistent A win when swapped (in BA order, A is candidate_2 so winner is "2")
    parsed_ba_agree = {
        "candidate_1": {"clarity": 2.0, "naturalness": 3.0, "meaning_preservation": 4.0},
        "candidate_2": {"clarity": 4.0, "naturalness": 5.0, "meaning_preservation": 5.0},
        "winner": "2",
        "rationale": "Candidate 2 reads better.",
    }
    res_agree = reconcile_swapped_verdicts(parsed_ab, parsed_ba_agree, sample_id="s1")
    assert res_agree.winner == "a"
    assert res_agree.agreed is True
    assert res_agree.verdict_ab == "a" and res_agree.verdict_ba == "a"
    assert res_agree.scores_a == {"clarity": 4.5, "naturalness": 4.5, "meaning_preservation": 5.0}
    assert res_agree.scores_b == {"clarity": 2.0, "naturalness": 3.0, "meaning_preservation": 4.0}

    # Position bias: judge picks "1" in both AB and BA -> must strictly reconcile to "tie"
    parsed_ba_bias = {
        "candidate_1": {"clarity": 4.0, "naturalness": 4.0, "meaning_preservation": 4.0},
        "candidate_2": {"clarity": 3.0, "naturalness": 3.0, "meaning_preservation": 3.0},
        "winner": "1",
        "rationale": "First candidate preferred.",
    }
    res_bias = reconcile_swapped_verdicts(parsed_ab, parsed_ba_bias, sample_id="s2")
    assert res_bias.winner == "tie"
    assert res_bias.agreed is False
    assert res_bias.verdict_ab == "a" and res_bias.verdict_ba == "b"


def test_blind_pairwise_judge_http_mock_position_swap_and_retry():
    call_count = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        # Simulate a transient 429 on the very first request to verify retry/backoff
        if call_count == 1:
            return httpx.Response(429, json={"error": "rate_limited"})

        body = json.loads(request.content.decode("utf-8"))
        user_msg = body["messages"][1]["content"]

        # Pair 1 ("clean_a" vs "dense_b"): consistently prefers "clean_a" regardless of slot
        if "clean_a" in user_msg:
            c1_is_clean = user_msg.index("clean_a") < user_msg.index("dense_b")
            if c1_is_clean:
                verdict = {
                    "candidate_1": {"clarity": 5, "naturalness": 5, "meaning_preservation": 5},
                    "candidate_2": {"clarity": 2, "naturalness": 2, "meaning_preservation": 4},
                    "winner": "1",
                    "rationale": "Candidate 1 is cleaner.",
                }
            else:
                verdict = {
                    "candidate_1": {"clarity": 2, "naturalness": 2, "meaning_preservation": 4},
                    "candidate_2": {"clarity": 5, "naturalness": 5, "meaning_preservation": 5},
                    "winner": "2",
                    "rationale": "Candidate 2 is cleaner.",
                }
        else:
            # Pair 2: exhibits pure position bias (always picks Candidate 1) -> must reconcile to tie
            verdict = {
                "candidate_1": {"clarity": 4, "naturalness": 4, "meaning_preservation": 4},
                "candidate_2": {"clarity": 3, "naturalness": 3, "meaning_preservation": 3},
                "winner": "1",
                "rationale": "Picked first slot.",
            }

        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(verdict)}}]},
        )

    transport = httpx.MockTransport(_handler)
    client = httpx.Client(transport=transport)
    judge = BlindPairwiseJudge(
        model="openai/gpt-4o-mini",
        provider="openrouter",
        api_key="test-key",
        max_retries=2,
        backoff_base=0.0,
        max_concurrency=2,
        client=client,
    )

    pairs = [
        ("p1", "Original dense text.", "clean_a rewrite.", "dense_b rewrite."),
        ("p2", "Another original.", "cand_x rewrite.", "cand_y rewrite."),
    ]
    summary = judge.evaluate_pairs(pairs, method_a="LoRA", method_b="Original")
    client.close()

    assert isinstance(summary, PairwiseSummary)
    assert summary.total_pairs == 2
    assert summary.wins_a == 1
    assert summary.wins_b == 0
    assert summary.ties == 1
    assert summary.win_rate_a == 0.5
    assert summary.tie_rate == 0.5
    assert summary.agreement_rate == 0.5
    assert summary.results[0].sample_id == "p1" and summary.results[0].winner == "a"
    assert summary.results[1].sample_id == "p2" and summary.results[1].winner == "tie"
    assert call_count == 5  # 1 transient 429 + 4 successful swapped evaluations


def test_summarize_pairwise_empty():
    empty = summarize_pairwise([], method_a="M1", method_b="M2")
    assert empty.total_pairs == 0
    assert empty.win_rate_a == 0.0
    assert empty.agreement_rate == 0.0


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


def test_evaluate_jsonl_loader_and_out_of_lexicon_breakdown(tmp_path, monkeypatch):
    # Stub heavy NLP modules only if absent in minimal CPU test environments
    for mod_name in ("spacy", "nltk", "nltk.corpus"):
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
            except ImportError:
                stub = ModuleType(mod_name)
                if mod_name == "nltk.corpus":
                    stub.brown = None
                    stub.reuters = None
                    stub.wordnet = None
                monkeypatch.setitem(sys.modules, mod_name, stub)

    from psycho_rev.evaluate import (
        _append_main_table,
        _append_out_of_lexicon_table,
        _summarize_out_of_lexicon,
        _summarize_rows,
        load_jsonl_dataset,
    )

    jsonl_path = tmp_path / "real_world_test.jsonl"
    records = [
        {
            "sample_id": "rw_01",
            "domain": "technical_blog",
            "text": "The service mesh proxy routing rule engineers configured stalled.",
            "is_out_of_lexicon": True,
        },
        {
            "sample_id": "rw_02",
            "domain": "essay",
            "text": "The committee reviewed the quarterly budget proposal.",
            "is_out_of_lexicon": False,
        },
        {
            "sample_id": "hc_01",
            "domain": "human_control",
            "text": "We fixed the bug before the morning release.",
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    test_samples, control_samples = load_jsonl_dataset(jsonl_path)
    assert len(test_samples) == 2
    assert len(control_samples) == 1
    assert test_samples[0].is_out_of_lexicon is True
    assert test_samples[1].is_out_of_lexicon is False

    mock_rows = [
        {
            "difficulty_score": 42.0,
            "unattested_bigram_rate": 0.25,
            "trigram_perplexity": 180.0,
            "mean_dep_length": 3.1,
            "reduced_orc_per_100w": 2.0,
            "stacked_noun_per_100w": 3.0,
            "denominal_verb_per_100w": 1.0,
            "inanimate_subj_per_100w": 1.0,
            "cosine_similarity": 0.92,
            "nli_min_entailment": 0.85,
            "nli_max_contradiction": 0.03,
            "gpt2_perplexity": 35.0,
            "acceptability_score": 0.62,
            "acceptability_mean": 0.78,
            "acceptability_min": 0.62,
            "code_preservation": 1.0,
            "length_ratio": 1.02,
            "latency_ms": 12.0,
            "is_out_of_lexicon": True,
        },
        {
            "difficulty_score": 18.0,
            "unattested_bigram_rate": 0.10,
            "trigram_perplexity": 90.0,
            "mean_dep_length": 2.3,
            "reduced_orc_per_100w": 0.0,
            "stacked_noun_per_100w": 0.0,
            "denominal_verb_per_100w": 0.0,
            "inanimate_subj_per_100w": 0.0,
            "cosine_similarity": 0.96,
            "nli_min_entailment": 0.93,
            "nli_max_contradiction": 0.01,
            "gpt2_perplexity": 21.0,
            "acceptability_score": 0.94,
            "acceptability_mean": 0.96,
            "acceptability_min": 0.94,
            "code_preservation": 1.0,
            "length_ratio": 1.00,
            "latency_ms": 8.0,
            "is_out_of_lexicon": False,
        },
    ]

    summary = _summarize_rows(mock_rows)
    assert math.isclose(summary["acceptability_mean"], 0.87, abs_tol=1e-4)
    assert math.isclose(summary["acceptability_min"], 0.78, abs_tol=1e-4)

    ool = _summarize_out_of_lexicon(mock_rows)
    assert ool["out_of_lexicon"]["count"] == 1
    assert ool["in_lexicon"]["count"] == 1
    assert math.isclose(ool["out_of_lexicon"]["acceptability_min"], 0.62, abs_tol=1e-4)
    assert math.isclose(ool["in_lexicon"]["acceptability_min"], 0.94, abs_tol=1e-4)

    md_lines: list[str] = []
    _append_main_table(md_lines, "## Main", [{"method": "Test", **summary}], None, "Ref")
    _append_out_of_lexicon_table(md_lines, "### OOL", [{"method": "Test", **ool}])
    rendered = "\n".join(md_lines)
    assert "Acc Mean ↑" in rendered and "Acc Min ↑" in rendered
    assert "OOL Acc Min ↑" in rendered
def test_split_prose_sentences_strips_frontmatter_tables_headings_and_joins_wrapped_lines():
    markdown_doc = (
        "---\n"
        "title: Deployment Guide\n"
        "owner: platform-team\n"
        "tags: [ops, rollout]\n"
        "---\n"
        "# Service Overview\n\n"
        "The ingestion pipeline validates incoming payloads\n"
        "before writing records to durable storage.\n"
        "Operators monitor queue depth during peak traffic\n"
        "to avoid cascading timeouts.\n\n"
        "| Stage | Timeout | Owner |\n"
        "| :--- | ---: | :--- |\n"
        "| canary | 60s | sre |\n"
        "| prod | 120s | core |\n\n"
        "## Rollback Procedure\n"
        "Trigger the automated rollback workflow when\n"
        "error budgets drop below the threshold.\n"
        "- Verify health checks pass.\n"
        "- Confirm alert noise subsides."
    )

    sents = split_prose_sentences(markdown_doc)

    assert all("Deployment Guide" not in s and "platform-team" not in s for s in sents)
    assert all("|" not in s and "canary" not in s and "60s" not in s for s in sents)
    assert all(not s.startswith("#") for s in sents)
    assert sents == [
        "Service Overview",
        "The ingestion pipeline validates incoming payloads before writing records to durable storage.",
        "Operators monitor queue depth during peak traffic to avoid cascading timeouts.",
        "Rollback Procedure",
        "Trigger the automated rollback workflow when error budgets drop below the threshold.",
        "Verify health checks pass.",
        "Confirm alert noise subsides.",
    ]


def test_evaluate_dynamic_four_domain_summary_and_plot(tmp_path, monkeypatch):
    for mod_name in ("spacy", "nltk", "nltk.corpus"):
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
            except ImportError:
                stub = ModuleType(mod_name)
                if mod_name == "nltk.corpus":
                    stub.brown = None
                    stub.reuters = None
                    stub.wordnet = None
                monkeypatch.setitem(sys.modules, mod_name, stub)

    import matplotlib.pyplot as plt
    from psycho_rev.evaluate import (
        _append_domain_table,
        _plot_comparison,
        _summarize_by_domain,
        _summarize_rows,
    )

    domain_specs = {
        "coding": (31.5, 0.88, 0.91),
        "technical_docs": (44.2, 0.84, 0.86),
        "essay": (27.8, 0.92, 0.95),
        "workplace_chat": (19.4, 0.89, 0.93),
    }

    rows_orig = [
        {
            "domain": dom,
            "difficulty_score": diff,
            "unattested_bigram_rate": 0.2,
            "trigram_perplexity": 120.0,
            "mean_dep_length": 2.8,
            "reduced_orc_per_100w": 1.0,
            "stacked_noun_per_100w": 2.0,
            "denominal_verb_per_100w": 0.5,
            "inanimate_subj_per_100w": 0.5,
            "cosine_similarity": 0.95,
            "nli_min_entailment": nli,
            "nli_max_contradiction": 0.02,
            "gpt2_perplexity": 28.0,
            "acceptability_score": acc_min,
            "acceptability_mean": acc_min + 0.02,
            "acceptability_min": acc_min,
            "code_preservation": 1.0,
            "length_ratio": 1.0,
            "latency_ms": 10.0,
        }
        for dom, (diff, nli, acc_min) in domain_specs.items()
    ]

    summary_dense = [{"method": "Original", **_summarize_rows(rows_orig)}]
    domain_breakdowns = [{"method": "Original", **_summarize_by_domain(rows_orig)}]

    md_lines: list[str] = []
    _append_domain_table(md_lines, "### Per-Domain Breakdown", domain_breakdowns)
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    rendered = summary_path.read_text(encoding="utf-8")

    assert "Chat Diff ↓" not in rendered
    for dom, (diff, nli, acc_min) in domain_specs.items():
        assert f"{dom} Diff ↓" in rendered
        assert f"{dom} NLI ↑" in rendered
        assert f"{dom} Acc Min ↑" in rendered
        assert f"{diff:.1f}" in rendered
        assert f"{nli:.3f}" in rendered
        assert f"{acc_min:.3f}" in rendered

    captured_legend_labels: list[str] = []
    captured_bar_heights: list[float] = []
    orig_close = plt.close

    def _inspect_and_close(fig=None):
        if fig is not None and hasattr(fig, "axes") and len(fig.axes) >= 2:
            ax_dom = fig.axes[1]
            legend = ax_dom.get_legend()
            if legend is not None:
                captured_legend_labels.extend(t.get_text() for t in legend.get_texts())
            captured_bar_heights.extend(patch.get_height() for patch in ax_dom.patches)
        return orig_close(fig)

    monkeypatch.setattr(plt, "close", _inspect_and_close)
    plot_path = tmp_path / "comparison_plot.png"
    _plot_comparison(
        summaries_dense=summary_dense,
        domain_breakdowns=domain_breakdowns,
        summaries_multipara=[],
        human_ref=None,
        out_path=plot_path,
    )
    assert plot_path.exists() and plot_path.stat().st_size > 0
    assert set(captured_legend_labels) == {
        "Coding",
        "Technical Docs",
        "Essay",
        "Workplace Chat",
    }
    assert len(captured_bar_heights) == 4
    assert all(h > 0.0 for h in captured_bar_heights)
