from psycho_rev.features import (
    CACHE_DIR,
    PsycholinguisticAnalyzer,
    ReferenceNgramModel,
    find_stacked_noun_spans,
)


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


def test_stacked_noun_spans_respect_np_boundary_in_essay_05():
    analyzer = PsycholinguisticAnalyzer()
    text = "The carbon border adjustment mechanism trade ministers negotiated bites hardest"
    doc = analyzer.nlp(text)
    spans = find_stacked_noun_spans(doc)
    span_texts = [s[2] for s in spans]
    assert "carbon border adjustment mechanism" in span_texts
    assert "carbon border adjustment mechanism trade ministers" not in span_texts
    # Verify (start_idx, end_idx_inclusive, span_text) indexing invariant
    for start_i, end_i, span_text in spans:
        assert doc[start_i : end_i + 1].text == span_text


def test_fenced_code_blocks_excluded_from_prose_feature_counts():
    analyzer = PsycholinguisticAnalyzer()
    prose = "The library closes at six on Sundays, so return the books in the morning."
    with_code = (
        prose
        + "\n\n```python\n"
        + "# The retry backoff timer config chokes and bottlenecks the pipeline\n"
        + "def sunset_legacy_workflow():\n"
        + "    return parse_header()\n"
        + "```\n"
    )
    r_prose = analyzer.analyze(prose)
    r_code = analyzer.analyze(with_code)
    assert r_code.word_count == r_prose.word_count
    assert r_code.stacked_noun_count == r_prose.stacked_noun_count
    assert r_code.denominal_verb_count == r_prose.denominal_verb_count
    assert r_code.inanimate_subj_count == r_prose.inanimate_subj_count


def test_reference_ngram_model_v2_includes_modern_tech_vocab():
    model = ReferenceNgramModel.load_or_build()
    assert (CACHE_DIR / "ngram_ref_v2.pkl").exists()
    for term in ("timezone", "monorepo", "endianness", "kubernetes", "middleware", "webhook"):
        assert model.unigrams[term] > 0
