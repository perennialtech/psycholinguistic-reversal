from psycho_rev.benchmark_data import HUMAN_CONTROL
from psycho_rev.classical import ClassicalReverser
from psycho_rev.code_guard import code_preservation_rate
from psycho_rev.features import PsycholinguisticAnalyzer


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


def test_stacked_noun_unpacking_uses_natural_prepositions_and_respects_boundaries():
    reverser = ClassicalReverser()

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
    assert "```python\n# `is_file()` swallows errors and sunsetted roles\ndef sunset():\n    return Pool.acquire()\n```" in out
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
