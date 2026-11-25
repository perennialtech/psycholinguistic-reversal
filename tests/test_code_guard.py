from psycho_rev.code_guard import code_preservation_rate, protect_spans


def test_protect_and_restore_code_spans():
    text = "Right now `is_file()` swallows errors in `src/cache/store.py` and calls parse_header()."
    guarded = protect_spans(text)
    assert "`is_file()`" not in guarded.masked
    assert "`src/cache/store.py`" not in guarded.masked
    assert "parse_header()" not in guarded.masked
    assert guarded.restore(guarded.masked) == text
    assert code_preservation_rate(text, text) == 1.0
    assert code_preservation_rate(text, "Nothing preserved here.") == 0.0
