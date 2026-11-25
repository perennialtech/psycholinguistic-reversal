from psycho_rev.code_guard import (
    code_preservation_rate,
    extract_protected_tokens,
    protect_spans,
    rewrite_markdown_preserving_structure,
)


def test_protect_and_restore_code_spans():
    text = "Right now `is_file()` swallows errors in `src/cache/store.py` and calls parse_header()."
    guarded = protect_spans(text)
    assert "`is_file()`" not in guarded.masked
    assert "`src/cache/store.py`" not in guarded.masked
    assert "parse_header()" not in guarded.masked
    assert guarded.restore(guarded.masked) == text
    assert code_preservation_rate(text, text) == 1.0
    assert code_preservation_rate(text, "Nothing preserved here.") == 0.0


def test_restore_handles_more_than_ten_placeholders_without_prefix_collision():
    items = [f"`fn_{i}()`" for i in range(13)]
    text = "Call " + ", ".join(items) + " in order."
    guarded = protect_spans(text)
    assert "CODEITEM1" in guarded.placeholders
    assert "CODEITEM10" in guarded.placeholders
    assert "CODEITEM11" in guarded.placeholders
    assert "CODEITEM12" in guarded.placeholders
    restored = guarded.restore(guarded.masked)
    assert restored == text
    assert "CODEITEM" not in restored


def test_protects_fenced_blocks_dotted_identifiers_and_bare_filenames():
    text = (
        "See Pool.acquire(), Pool.acquire, and os.path.join in store.py, "
        "routes.yaml, mutex.rs, Cargo.toml, and package.json (e.g. for config).\n\n"
        "```python\n"
        "def run():\n"
        "    return Pool.acquire()\n"
        "```\n\n"
        "Check https://example.com/docs/v2 and src/cache/store.py for details."
    )
    guarded = protect_spans(text)
    tokens = extract_protected_tokens(text)
    for expected in (
        "Pool.acquire()",
        "Pool.acquire",
        "os.path.join",
        "store.py",
        "routes.yaml",
        "mutex.rs",
        "Cargo.toml",
        "package.json",
        "https://example.com/docs/v2",
        "src/cache/store.py",
    ):
        assert expected in tokens
        assert expected not in guarded.masked

    # Fenced code block protected verbatim
    assert "def run():" not in guarded.masked
    # Ordinary prose abbreviation and punctuation remain intact
    assert "e.g." in guarded.masked
    assert guarded.restore(guarded.masked) == text


def test_ordinary_prose_punctuation_and_slashes_not_masked():
    text = "Use clear names (e.g. user_count) and/or short sentences. Next, check the U.S. time zone."
    guarded = protect_spans(text)
    assert "e.g." in guarded.masked
    assert "and/or" in guarded.masked
    assert "U.S." in guarded.masked
    assert guarded.restore(guarded.masked) == text


def test_rewrite_markdown_preserving_structure_single_line_matches_rewrite_fn():
    plain = "The connection pool fails under heavy load."
    fn = lambda s: s.upper()
    assert rewrite_markdown_preserving_structure(plain, fn) == fn(plain)


def test_rewrite_markdown_preserving_structure_multiline():
    doc = (
        "# Architecture Notes\n"
        "\n"
        "First paragraph text.\n"
        "\n"
        "- Bullet one\n"
        "* Bullet two\n"
        "1. Numbered item\n"
        "\n"
        "```python\n"
        "# do not touch this\n"
        "x = Pool.acquire()\n"
        "```\n"
        "\n"
        "Closing paragraph.\n"
    )
    out = rewrite_markdown_preserving_structure(doc, lambda s: f"[{s}]")
    lines = out.splitlines()
    assert lines[0] == "# [Architecture Notes]"
    assert lines[1] == ""
    assert lines[2] == "[First paragraph text.]"
    assert lines[3] == ""
    assert lines[4] == "- [Bullet one]"
    assert lines[5] == "* [Bullet two]"
    assert lines[6] == "1. [Numbered item]"
    assert lines[7] == ""
    assert lines[8] == "```python"
    assert lines[9] == "# do not touch this"
    assert lines[10] == "x = Pool.acquire()"
    assert lines[11] == "```"
    assert lines[12] == ""
    assert lines[13] == "[Closing paragraph.]"
    assert out.endswith("\n")
