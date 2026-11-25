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
