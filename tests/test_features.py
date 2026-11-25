from psycho_rev.features import PsycholinguisticAnalyzer


def test_detects_canonical_paper_sentence():
    analyzer = PsycholinguisticAnalyzer()
    dense = (
        "Your instinct is right; the convoluted sentence you are currently reading "
        "is quietly eating its own reader-patience budget — footnote it."
    )
    rep = analyzer.analyze(dense)
    assert rep.reduced_orc_count >= 1
    assert rep.stacked_noun_count >= 1
    assert rep.denominal_verb_count >= 1
    assert rep.inanimate_subj_count >= 1
    assert rep.difficulty_score > 45.0


def test_human_sentence_scores_lower_than_dense_ai():
    analyzer = PsycholinguisticAnalyzer()
    dense = (
        "The connection pool manager chokes under burst traffic because the retry backoff "
        "timer config you added starves worker threads and bottlenecks the pipeline."
    )
    simple = (
        "The connection pool fails under heavy traffic because the retry timer that you "
        "added leaves worker threads without resources and slows down the process."
    )
    r_dense = analyzer.analyze(dense)
    r_simple = analyzer.analyze(simple)
    assert r_simple.difficulty_score < r_dense.difficulty_score
