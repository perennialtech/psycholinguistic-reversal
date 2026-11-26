import json
from types import SimpleNamespace

import httpx

from psycho_rev.benchmark_data import HUMAN_CONTROL
from psycho_rev.classical import ClassicalReverser
from psycho_rev.code_guard import code_preservation_rate
from psycho_rev.features import PsycholinguisticAnalyzer
from psycho_rev.neural import (
    REWRITE_SYSTEM_PROMPT,
    BestOfNPsycholinguisticReverser,
    HybridReverser,
    OpenAICompatibleReverser,
)


def test_classical_reverser_reduces_features_and_preserves_code():
    analyzer = PsycholinguisticAnalyzer()
    reverser = ClassicalReverser()
    text = (
        "Right now `is_file()` swallows that permission error, which quietly sabotages "
        "the cache invalidation pipeline you added in `src/cache/store.py`."
    )
    res = reverser.rewrite_with_stats(text)
    assert code_preservation_rate(text, res.text) == 1.0
    before = analyzer.analyze(text)
    after = analyzer.analyze(res.text)
    assert after.inanimate_subj_count < before.inanimate_subj_count
    assert after.difficulty_score < before.difficulty_score


def test_relative_adverb_not_treated_as_reduced_orc():
    reverser = ClassicalReverser()
    text = "There are not many places where we can talk about important things together."
    out = reverser.rewrite(text)
    assert "that where" not in out
    assert out == text


def test_safe_mode_disables_brittle_noun_inversion_by_default():
    safe_reverser = ClassicalReverser()
    assert safe_reverser.safe_mode is True
    assert safe_reverser.allow_noun_inversion is False

    tricky = "The quarterly roadmap alignment deck product leadership circulated needs review."
    res = safe_reverser.rewrite_with_stats(tricky)
    assert res.edits_applied["noun_stack_unpacked"] == 0
    assert "deck for quarterly roadmap alignment" not in res.text
    assert "leadership for" not in res.text


def test_stacked_noun_unpacking_uses_natural_prepositions_in_legacy_mode():
    reverser = ClassicalReverser(safe_mode=False)

    s_boundary = (
        "The serialization format quietly betrays endianness assumptions "
        "whenever `read_u64()` encounters an unaligned memory page boundary."
    )
    out_boundary = reverser.rewrite(s_boundary)
    assert "boundary of memory page" in out_boundary
    assert "boundary for memory page" not in out_boundary

    s_essay05 = "The carbon border adjustment mechanism trade ministers negotiated bites hardest."
    out_essay05 = reverser.rewrite(s_essay05)
    assert "mechanism for carbon border adjustment" in out_essay05
    assert "ministers for" not in out_essay05


def test_fluency_gate_reverts_degrading_sentence_edits_only():
    # Reject any sentence containing "silently ignores", accept "phase out"
    def fake_fluency(sent: str) -> float:
        if "silently ignores" in sent:
            return 0.40
        return 0.90

    reverser = ClassicalReverser(
        safe_mode=True,
        fluency_gate=fake_fluency,
        min_fluency_delta=-0.05,
    )
    text = (
        "Right now `is_file()` swallows errors in store.py. "
        "We should sunset `v1/auth.py` next quarter."
    )
    res = reverser.rewrite_with_stats(text)
    # First sentence edit degraded fluency -> automatically reverted
    assert "Right now `is_file()` swallows errors in store.py." in res.text
    # Second sentence edit kept fluency -> retained
    assert "We should phase out `v1/auth.py` next quarter." in res.text
    assert res.edits_applied["fluency_reverted"] == 1
    assert res.edits_applied["inanimate_verb_replaced"] == 0
    assert res.edits_applied["denominal_expanded"] == 1


def test_passive_denominal_verb_inflection():
    reverser = ClassicalReverser()
    text = (
        "Port automation labor agreements will need to spotlight retraining pathways "
        "before legacy dockyard roles are sunsetted."
    )
    out = reverser.rewrite(text)
    assert "are phased out" in out
    assert "sunsetted" not in out


def test_context_gated_literal_denominals_not_mangled():
    reverser = ClassicalReverser()
    literal_cases = [
        "The submarine surfaced near the harbor.",
        "The ship anchored in the bay before dusk.",
        "The intern shadowed the senior engineer all week.",
    ]
    for sentence in literal_cases:
        assert reverser.rewrite(sentence) == sentence


def test_multiline_markdown_structure_preserved_during_rewrite():
    reverser = ClassicalReverser()
    doc = (
        "# Migration Plan\n"
        "\n"
        "Before we sunset `v1/auth.py`, we should signpost the migration steps.\n"
        "\n"
        "- Right now `is_file()` swallows errors in store.py.\n"
        "1. We should benchmark the cold-start latency overhead.\n"
        "\n"
        "```python\n"
        "# `is_file()` swallows errors and sunsetted roles\n"
        "def sunset():\n"
        "    return Pool.acquire()\n"
        "```\n"
    )
    res = reverser.rewrite_with_stats(doc)
    out = res.text
    assert out.startswith("# Migration Plan\n\n")
    assert "phase out `v1/auth.py`" in out
    assert "- Right now `is_file()` silently ignores errors in store.py." in out
    assert "1. We should measure" in out
    assert (
        "```python\n# `is_file()` swallows errors and sunsetted roles\ndef sunset():\n    return Pool.acquire()\n```"
        in out
    )
    assert code_preservation_rate(doc, out) == 1.0


def test_human_control_sentences_remain_untouched_or_undamaged():
    reverser = ClassicalReverser()
    analyzer = PsycholinguisticAnalyzer()
    for sample in HUMAN_CONTROL:
        res = reverser.rewrite_with_stats(sample.text)
        assert code_preservation_rate(sample.text, res.text) == 1.0
        assert res.text == sample.text, f"Unexpected edit on {sample.sample_id}: {res.text!r}"
        before = analyzer.analyze(sample.text)
        after = analyzer.analyze(res.text)
        assert after.difficulty_score <= before.difficulty_score + 0.01


def test_rewrite_system_prompt_is_deleaked():
    lower = REWRITE_SYSTEM_PROMPT.lower()
    for leaked_word in (
        "footnote",
        "calendar",
        "triage",
        "workshop",
        "swallows",
        "sabotages",
        "starves",
        "punishes",
    ):
        assert leaked_word not in lower
    assert "CODEITEM" in REWRITE_SYSTEM_PROMPT
    assert "unchanged" in lower


def test_openai_compatible_reverser_rewrite_retry_batch_and_clarity_gate():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        # Simulate one transient 429 before succeeding
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "rate_limited"})
        user_msg = body["messages"][-1]["content"]
        # Echo back a simplified sentence preserving placeholders
        if "CODEITEM0" in user_msg:
            content = "The helper CODEITEM0 ignores errors in CODEITEM1."
        else:
            content = "Simplified clear prose."
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": content}}]},
        )

    class _StubAnalyzer:
        def analyze(self, text: str) -> SimpleNamespace:
            # Treat short/simple strings as already clear (score 20.0), dense text as 60.0
            score = 20.0 if "already clear" in text else 60.0
            return SimpleNamespace(difficulty_score=score)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reverser = OpenAICompatibleReverser(
        model="test-model",
        base_url="https://api.example.test/v1",
        api_key="sk-test",
        max_retries=2,
        retry_backoff=0.0,
        max_concurrency=2,
        skip_if_clear=True,
        clear_difficulty_threshold=35.0,
        analyzer=_StubAnalyzer(),
        client=client,
    )

    # 1. Clarity no-op gate skips HTTP call completely
    clear_text = "This sentence is already clear and readable."
    assert reverser.rewrite(clear_text) == clear_text
    assert len(calls) == 0

    # 2. Dense text retries past 429 and preserves code spans
    dense_text = "Right now `is_file()` swallows errors in `store.py`."
    rewritten = reverser.rewrite(dense_text)
    assert rewritten == "The helper `is_file()` ignores errors in `store.py`."
    assert code_preservation_rate(dense_text, rewritten) == 1.0
    assert len(calls) == 2  # 1 failed 429 + 1 succeeded

    # 3. rewrite_batch preserves order with bounded concurrency
    batch_inputs = [
        "This sentence is already clear.",
        "Dense paragraph one needs work.",
        "Dense paragraph two needs work.",
    ]
    batch_out = reverser.rewrite_batch(batch_inputs, max_concurrency=2)
    assert batch_out == [
        "This sentence is already clear.",
        "Simplified clear prose.",
        "Simplified clear prose.",
    ]
    client.close()


def test_openai_compatible_reverser_falls_back_on_code_loss():
    def handler(request: httpx.Request) -> httpx.Response:
        # Always drop placeholders to trigger fallback
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Dropped all code."}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reverser = OpenAICompatibleReverser(
        model="test-model",
        base_url="http://localhost:11434/v1",
        skip_if_clear=False,
        fallback_on_code_loss=True,
        client=client,
    )
    src = "Call `Pool.acquire()` in `store.py`."
    assert reverser.rewrite(src) == src
    client.close()


def test_best_of_n_penalizes_acceptability_and_enforces_nli_floor_and_clarity_gate():
    class _FakeGen:
        def __init__(self):
            self.called = 0

        def generate_candidates(self, text: str, **kwargs) -> list[str]:
            self.called += 1
            return [
                "Bad grammar candidate `store.py`.",
                "Low NLI candidate `store.py`.",
                "Natural faithful rewrite `store.py`.",
            ]

    class _FakeAnalyzer:
        def analyze(self, text: str) -> SimpleNamespace:
            if "already clear" in text:
                return SimpleNamespace(difficulty_score=20.0)
            if "Bad grammar" in text:
                return SimpleNamespace(difficulty_score=5.0)
            if "Low NLI" in text:
                return SimpleNamespace(difficulty_score=5.0)
            if "Natural faithful" in text:
                return SimpleNamespace(difficulty_score=28.0)
            return SimpleNamespace(difficulty_score=65.0)

    class _FakeSemantics:
        def evaluate_batch(self, originals: list[str], candidates: list[str]) -> list[SimpleNamespace]:
            out = []
            for c in candidates:
                if "Bad grammar" in c:
                    out.append(
                        SimpleNamespace(
                            cosine_similarity=0.90,
                            nli_min_entailment=0.85,
                            nli_max_contradiction=0.02,
                            gpt2_perplexity=40.0,
                            code_preservation=1.0,
                            length_ratio=1.0,
                            acceptability_min=0.25,
                        )
                    )
                elif "Low NLI" in c:
                    out.append(
                        SimpleNamespace(
                            cosine_similarity=0.88,
                            nli_min_entailment=0.30,
                            nli_max_contradiction=0.10,
                            gpt2_perplexity=35.0,
                            code_preservation=1.0,
                            length_ratio=1.0,
                            acceptability_min=0.95,
                        )
                    )
                else:
                    out.append(
                        SimpleNamespace(
                            cosine_similarity=0.92,
                            nli_min_entailment=0.88,
                            nli_max_contradiction=0.02,
                            gpt2_perplexity=30.0,
                            code_preservation=1.0,
                            length_ratio=1.05,
                            acceptability_min=0.96,
                        )
                    )
            return out

    gen = _FakeGen()
    bon = BestOfNPsycholinguisticReverser(
        generator=gen,
        analyzer=_FakeAnalyzer(),
        semantics=_FakeSemantics(),
        skip_if_clear=True,
        clear_difficulty_threshold=35.0,
    )

    # Clarity gate skips generator
    assert bon.rewrite("This text is already clear.") == "This text is already clear."
    assert gen.called == 0

    # Reranker picks natural faithful candidate over low-NLI and low-acceptability metric gamers
    chosen = bon.rewrite("Dense source sentence mentioning `store.py`.")
    assert chosen == "Natural faithful rewrite `store.py`."
    assert gen.called == 1


def test_hybrid_reverser_defaults_to_safe_mode_and_supports_fluency_gate():
    class _IdentityNeural:
        def rewrite(self, text: str) -> str:
            return text

    hybrid = HybridReverser(_IdentityNeural())
    assert hybrid.classical.safe_mode is True
    assert hybrid.classical.allow_noun_inversion is False

    tricky = "The quarterly roadmap alignment deck product leadership circulated needs review."
    assert hybrid.rewrite(tricky) == tricky
