from psycho_rev.code_guard import (
    code_preservation_rate,
    extract_protected_tokens,
    protect_spans,
    restore_spans,
    rewrite_markdown_preserving_structure,
    split_markdown_chunks,
)


def test_protect_and_restore_code_spans():
    text = "Right now `is_file()` swallows errors in `src/cache/store.py` and calls parse_header()."
    guarded = protect_spans(text)
    assert "`is_file()`" not in guarded.masked
    assert "`src/cache/store.py`" not in guarded.masked
    assert "parse_header()" not in guarded.masked
    assert guarded.restore(guarded.masked) == text
    assert restore_spans(guarded.masked, guarded) == text
    assert restore_spans(guarded.masked, guarded.placeholders) == text
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


def test_protects_yaml_frontmatter_markdown_links_html_flags_and_env_vars():
    text = (
        "---\n"
        "title: Migration Guide\n"
        "tags: [infra, api]\n"
        "---\n\n"
        "Read the [migration guide](https://example.com/docs/v2_(beta)) and "
        "[local notes](./docs/setup.md#install) or [`is_file()`](https://example.com/api).\n"
        "Pass --dry-run and --max-tokens=512 with $HOME and ${OPENAI_API_KEY:-demo}.\n"
        'Keep <div class="note">important</div> and <br/> intact, while x < 5 and y > 3 stays prose, '
        "costing $100 -- really.\n"
    )
    guarded = protect_spans(text)
    tokens = extract_protected_tokens(text)

    # Frontmatter masked as a protected block
    assert "title: Migration Guide" not in guarded.masked
    # Markdown link labels stay intact while URL destinations are protected
    assert "[migration guide](" in guarded.masked
    assert "[local notes](" in guarded.masked
    assert "https://example.com/docs/v2_(beta)" in tokens
    assert "./docs/setup.md#install" in tokens
    assert "`is_file()`" in tokens
    assert "https://example.com/api" in tokens
    # CLI flags and environment variables protected
    assert "--dry-run" in tokens
    assert "--max-tokens=512" in tokens
    assert "$HOME" in tokens
    assert "${OPENAI_API_KEY:-demo}" in tokens
    # HTML tags protected
    assert '<div class="note">' in tokens
    assert "</div>" in tokens
    assert "<br/>" in tokens
    # Ordinary prose math, currency, and em-dash not masked
    assert "x < 5 and y > 3" in guarded.masked
    assert "$100" in guarded.masked
    assert " -- " in guarded.masked
    assert guarded.restore(guarded.masked) == text
    assert code_preservation_rate(text, text) == 1.0


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


def test_split_markdown_chunks_preserves_frontmatter_tables_blockquotes_and_respects_budget():
    long_para = " ".join(
        f"Sentence number {i} explains a distinct operational detail clearly." for i in range(1, 25)
    )
    doc = (
        "---\n"
        "title: Runbook\n"
        "owner: sre\n"
        "---\n\n"
        "# Operations\n\n"
        "> Quoted note about service health.\n\n"
        "| Service | Port |\n"
        "| --- | --- |\n"
        "| api | 8080 |\n\n"
        f"{long_para}\n\n"
        "  ~~~bash\n"
        "  echo $HOME --dry-run\n"
        "  ~~~\n"
    )

    chunks = split_markdown_chunks(doc, max_chars=250)
    assert chunks[0].kind == "frontmatter"
    assert chunks[0].rewritable is False
    assert any(c.kind == "table" and not c.rewritable for c in chunks)
    assert any(c.kind == "fence" and not c.rewritable for c in chunks)
    assert any(c.kind == "blockquote" and c.rewritable for c in chunks)

    para_chunks = [c for c in chunks if c.kind == "paragraph"]
    assert len(para_chunks) >= 3
    for pc in para_chunks:
        assert len(pc.text) <= 250

    seen_calls: list[str] = []

    def _recorder(s: str) -> str:
        seen_calls.append(s)
        return s

    rebuilt = rewrite_markdown_preserving_structure(doc, _recorder, max_chunk_chars=250)
    assert rebuilt == doc
    assert all(len(call) <= 250 for call in seen_calls)
    assert not any("title: Runbook" in call for call in seen_calls)
    assert not any("| api | 8080 |" in call for call in seen_calls)
    assert not any("echo $HOME --dry-run" in call for call in seen_calls)
