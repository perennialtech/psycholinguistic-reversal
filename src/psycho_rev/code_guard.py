"""Protect code blocks, inline code spans, markdown links, HTML/XML, CLI flags, env vars, URLs, identifiers, and paths during text rewriting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

_CODE_EXTENSIONS = (
    "py|pyi|pyx|js|mjs|cjs|ts|tsx|jsx|rs|go|cc|cpp|cxx|hpp|hh|"
    "java|kt|kts|scala|rb|php|pl|sh|bash|zsh|fish|ps1|bat|cmd|cs|fs|"
    "swift|dart|lua|jl|ex|exs|erl|hs|ml|zig|nim|sql|html|htm|css|"
    "scss|sass|less|xml|json|jsonc|jsonl|yaml|yml|toml|ini|cfg|conf|"
    "env|lock|md|rst|tex|csv|tsv|proto|graphql|gql|tf|hcl|nix|mk|"
    "cmake|gradle|txt|log"
)

_PROSE_SLASH_NEG = (
    r"(?!(?i:and|or|yes|no|on|off|he|she|his|her|him|either|neither|"
    r"true|false|in|out|up|down|input|output|read|write|client|server|"
    r"before|after|pros|cons)/)"
)

_YAML_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n[\s\S]*?\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)"
)

_FENCED_BLOCK_RE = re.compile(
    r"(?m)^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n[\s\S]*?^[ \t]*(?P=fence)[ \t]*$"
)

_PROTECTED_PATTERN = re.compile(
    r"("
    # 0. YAML frontmatter at the very start of the document
    r"\A---[ \t]*\r?\n[\s\S]*?\r?\n(?:---|\.\.\.)[ \t]*(?=\r?\n|\Z)"
    # 1. Fenced code blocks (triple+ backticks or tildes, preserving indentation/language tags)
    r"|(?m:^[ \t]*```[^\n]*\n[\s\S]*?^[ \t]*```[ \t]*$)"
    r"|(?m:^[ \t]*~~~[^\n]*\n[\s\S]*?^[ \t]*~~~[ \t]*$)"
    r"|```[\s\S]*?```"
    r"|~~~[\s\S]*?~~~"
    # 2. Inline backticks (double or single)
    r"|``[^`\n]+``"
    r"|`[^`\n]+`"
    # 3. Markdown link/image destinations inside [label](destination) without touching [label]
    r"|(?<=\]\()(?:<[^>\n]+>|(?:[^()\s\n]+|\([^()\n]*\))+(?:\s+\"[^\"\n]*\")?)(?=\))"
    # 4. HTML/XML comments and tags (avoiding math comparisons like a < b)
    r"|<!--[\s\S]*?-->"
    r"|</[A-Za-z][A-Za-z0-9:_-]*\s*>"
    r"|<[A-Za-z][A-Za-z0-9:_-]*(?:\s+[A-Za-z_:][A-Za-z0-9:._-]*(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'=<>`]+))?)*\s*/?>"
    # 5. URLs (excluding trailing sentence punctuation)
    r"|https?://[^\s<>()\[\]]+(?<![.,;:!?])"
    # 6. Environment variables ($VAR, ${VAR}, ${VAR:-default})
    r"|\$\{[A-Za-z_][A-Za-z0-9_]*(?:[^}\n]*)\}"
    r"|\$[A-Za-z_][A-Za-z0-9_]*\b"
    # 7. CLI long flags (--flag-name, --flag=value)
    r"|(?<![A-Za-z0-9_-])--[A-Za-z0-9][A-Za-z0-9_-]*(?:=[^\s,;)\]]+)?"
    # 8. Dotted module/method calls with parentheses (e.g. Pool.acquire(), os.path.join())
    r"|\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\(\)"
    # 9. Bare function calls (e.g. parse_header(), is_file())
    r"|\b[A-Za-z_][A-Za-z0-9_]*\(\)"
    # 10. Slash paths (e.g. src/cache/store.py, ./run.sh, /etc/hosts), excluding common prose pairs
    r"|(?<![A-Za-z0-9_./-])(?:(?:\.{1,2}/|/)[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)"
    r"|\b" + _PROSE_SLASH_NEG + r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+\b"
    # 11. Bare filenames with common code/config extensions (e.g. store.py, routes.yaml, Cargo.toml)
    r"|(?<![A-Za-z0-9_./-])\.?[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:" + _CODE_EXTENSIONS + r")\b"
    # 12. Dotted attributes / module paths without parens (segments >= 2 chars so e.g./i.e./U.S. stay prose)
    r"|\b[A-Za-z_][A-Za-z0-9_]+(?:\.[A-Za-z_][A-Za-z0-9_]+)+\b(?!\.)"
    r")"
)

_FENCE_OPEN_RE = re.compile(r"^([ \t]*)(`{3,}|~{3,})(.*)$")
_HEADING_RE = re.compile(r"^([ \t]*#{1,6}[ \t]+)(.*)$")
_LIST_ITEM_RE = re.compile(r"^([ \t]*(?:[-*+]|\d+[.)])[ \t]+(?:\[[ xX]\][ \t]+)?)(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^([ \t]*(?:>[ \t]*)+)(.*)$")
_HR_RE = re.compile(r"^[ \t]*(?:---+|\*\*\*+|___+)[ \t]*$")
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
_TABLE_SEP_RE = re.compile(r"^[ \t]*\|?(?:[ \t]*:?-{2,}:?[ \t]*\|)+[ \t]*:?-{2,}:?[ \t]*\|?[ \t]*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class GuardedText:
    masked: str
    placeholders: dict[str, str]

    def restore(self, rewritten: str) -> str:
        return restore_spans(rewritten, self)


@dataclass
class MarkdownChunk:
    """A structural unit produced by `split_markdown_chunks`."""

    kind: str
    text: str
    prefix: str = ""
    rewritable: bool = True
    join_with: str = "\n"


def restore_spans(rewritten: str, guarded: GuardedText | dict[str, str]) -> str:
    """Restore placeholder tokens back to their original protected spans in a single pass."""
    placeholders = guarded.placeholders if isinstance(guarded, GuardedText) else guarded
    if not placeholders:
        return rewritten
    # Match longer placeholder keys first (e.g. CODEITEM10 before CODEITEM1) in one pass
    pattern = re.compile(
        "|".join(re.escape(k) for k in sorted(placeholders, key=len, reverse=True))
    )
    return pattern.sub(lambda m: placeholders[m.group(0)], rewritten)


def strip_fenced_code_blocks(text: str) -> str:
    """Remove fenced code blocks so downstream prose analyzers do not count code as prose."""
    return re.sub(r"```[\s\S]*?```|~~~[\s\S]*?~~~", "", text)


def _select_placeholder_prefix(text: str) -> str:
    """Return a placeholder prefix guaranteed not to appear anywhere in *text*."""
    if "CODEITEM" not in text:
        return "CODEITEM"
    suffix_idx = 0
    while f"CODEITEMX{suffix_idx}_" in text:
        suffix_idx += 1
    return f"CODEITEMX{suffix_idx}_"


def protect_spans(text: str) -> GuardedText:
    """Replace frontmatter, code spans, fenced blocks, link URLs, HTML/XML tags, CLI flags, env vars, URLs, paths, and identifiers with placeholders."""
    prefix = _select_placeholder_prefix(text)
    placeholders: dict[str, str] = {}
    counter = 0

    def _repl(match: re.Match[str]) -> str:
        nonlocal counter
        original = match.group(0)
        key = f"{prefix}{counter}"
        counter += 1
        placeholders[key] = original
        return key

    masked = _PROTECTED_PATTERN.sub(_repl, text)
    return GuardedText(masked=masked, placeholders=placeholders)


def extract_protected_tokens(text: str) -> list[str]:
    """Return all code-like and technical spans in a passage for fidelity verification."""
    return [m.group(0) for m in _PROTECTED_PATTERN.finditer(text)]


def code_preservation_rate(original: str, rewritten: str) -> float:
    """Fraction of protected spans from original preserved verbatim in rewritten."""
    tokens = extract_protected_tokens(original)
    if not tokens:
        return 1.0
    kept = sum(1 for tok in tokens if tok in rewritten)
    return kept / len(tokens)


def _split_text_to_budget(
    text: str,
    max_chars: int = 1800,
    max_words: int | None = None,
) -> list[str]:
    """Split a long prose block into chunks respecting char/word budgets at sentence boundaries."""
    stripped = text.strip()
    if not stripped:
        return [text]

    def _within_budget(candidate: str) -> bool:
        if max_chars > 0 and len(candidate) > max_chars:
            return False
        if max_words is not None and max_words > 0 and len(candidate.split()) > max_words:
            return False
        return True

    if _within_budget(stripped):
        return [stripped]

    sentences = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(stripped) if s.strip()]
    if not sentences:
        return [stripped]

    chunks: list[str] = []
    current: list[str] = []

    def _flush() -> None:
        if current:
            chunks.append(" ".join(current))
            current.clear()

    for sent in sentences:
        if not _within_budget(sent):
            _flush()
            # Fall back to word-level splitting for an oversized single sentence
            words = sent.split()
            word_buf: list[str] = []
            for w in words:
                trial = " ".join(word_buf + [w])
                if word_buf and not _within_budget(trial):
                    chunks.append(" ".join(word_buf))
                    word_buf = [w]
                else:
                    word_buf.append(w)
            if word_buf:
                chunks.append(" ".join(word_buf))
            continue

        trial = " ".join(current + [sent])
        if current and not _within_budget(trial):
            _flush()
            current.append(sent)
        else:
            current.append(sent)

    _flush()
    return chunks or [stripped]


def _is_table_line(line: str, next_line: str | None = None) -> bool:
    """Heuristic check for markdown pipe table lines."""
    if _TABLE_ROW_RE.match(line) or _TABLE_SEP_RE.match(line):
        return True
    if "|" in line and next_line is not None and _TABLE_SEP_RE.match(next_line):
        return True
    return False


def split_markdown_chunks(
    text: str,
    max_chars: int = 1800,
    max_words: int | None = None,
) -> list[MarkdownChunk]:
    """Split a markdown or plain-text document into structure-preserving chunks.

    Preserves YAML frontmatter, fenced code blocks, markdown tables, horizontal rules,
    blank lines, headings, list items, and blockquotes. Long prose paragraphs are split
    cleanly at sentence boundaries to respect `max_chars` / `max_words` budgets.
    """
    if not text:
        return []

    chunks: list[MarkdownChunk] = []
    body_text = text

    # Preserve YAML frontmatter verbatim at document start
    fm_match = _YAML_FRONTMATTER_RE.match(body_text)
    if fm_match:
        fm_block = fm_match.group(0).rstrip("\r\n")
        chunks.append(MarkdownChunk(kind="frontmatter", text=fm_block, rewritable=False))
        body_text = body_text[fm_match.end() :]
        if not body_text:
            return chunks

    lines = body_text.splitlines()
    prose_buf: list[str] = []

    def _emit_rewritable(kind: str, prefix: str, body: str) -> None:
        if not body.strip():
            chunks.append(MarkdownChunk(kind=kind, prefix=prefix, text=body, rewritable=False))
            return
        parts = _split_text_to_budget(body, max_chars=max_chars, max_words=max_words)
        for idx, part in enumerate(parts):
            chunks.append(
                MarkdownChunk(
                    kind=kind,
                    prefix=prefix if idx == 0 else "",
                    text=part,
                    rewritable=True,
                    join_with="\n" if idx == 0 else " ",
                )
            )

    def _flush_prose() -> None:
        if not prose_buf:
            return
        paragraph = " ".join(part.strip() for part in prose_buf)
        prose_buf.clear()
        _emit_rewritable("paragraph", "", paragraph)

    in_fence = False
    fence_char = ""
    fence_len = 0
    fence_lines: list[str] = []
    in_table = False

    for i, line in enumerate(lines):
        if in_fence:
            fence_lines.append(line)
            stripped = line.strip()
            if stripped.startswith(fence_char * fence_len) and set(stripped) == {fence_char}:
                chunks.append(
                    MarkdownChunk(kind="fence", text="\n".join(fence_lines), rewritable=False)
                )
                fence_lines.clear()
                in_fence = False
                fence_char = ""
                fence_len = 0
            continue

        fence_match = _FENCE_OPEN_RE.match(line)
        if fence_match:
            _flush_prose()
            in_table = False
            fence_lines = [line]
            in_fence = True
            fence_marker = fence_match.group(2)
            fence_char = fence_marker[0]
            fence_len = len(fence_marker)
            continue

        if not line.strip():
            _flush_prose()
            in_table = False
            chunks.append(MarkdownChunk(kind="blank", text=line, rewritable=False))
            continue

        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if in_table:
            if "|" in line:
                chunks.append(MarkdownChunk(kind="table", text=line, rewritable=False))
                continue
            in_table = False

        if _is_table_line(line, next_line):
            _flush_prose()
            in_table = True
            chunks.append(MarkdownChunk(kind="table", text=line, rewritable=False))
            continue

        if _HR_RE.match(line):
            _flush_prose()
            chunks.append(MarkdownChunk(kind="hr", text=line, rewritable=False))
            continue

        matched_prefix = False
        for kind, pattern in (
            ("heading", _HEADING_RE),
            ("list_item", _LIST_ITEM_RE),
            ("blockquote", _BLOCKQUOTE_RE),
        ):
            m = pattern.match(line)
            if m:
                _flush_prose()
                _emit_rewritable(kind, m.group(1), m.group(2))
                matched_prefix = True
                break
        if matched_prefix:
            continue

        prose_buf.append(line)

    if in_fence and fence_lines:
        chunks.append(MarkdownChunk(kind="fence", text="\n".join(fence_lines), rewritable=False))
    _flush_prose()
    return chunks


def rewrite_markdown_preserving_structure(
    text: str,
    rewrite_fn: Callable[[str], str],
    max_chunk_chars: int = 1800,
    max_chunk_words: int | None = None,
) -> str:
    """Rewrite prose inside a markdown or plain-text document while preserving structure.

    YAML frontmatter, fenced code blocks, tables, horizontal rules, and blank lines are
    preserved verbatim. Leading markdown heading markers (`# `), list prefixes (`- `, `* `,
    `1. `), and blockquote prefixes (`> `) are preserved while their prose bodies are
    passed to `rewrite_fn`. Long prose paragraphs are chunked cleanly at sentence
    boundaries to respect `max_chunk_chars` / `max_chunk_words`.
    Single-line plain text without markdown markers within budget behaves identically to
    `rewrite_fn(text)`.
    """
    if not text:
        return text

    if "\n" not in text and (max_chunk_chars <= 0 or len(text) <= max_chunk_chars) and (
        max_chunk_words is None or len(text.split()) <= max_chunk_words
    ):
        if _FENCE_OPEN_RE.match(text) or _HR_RE.match(text) or _TABLE_ROW_RE.match(text):
            return text
        for pattern in (_HEADING_RE, _LIST_ITEM_RE, _BLOCKQUOTE_RE):
            m = pattern.match(text)
            if m:
                prefix, body = m.group(1), m.group(2)
                return prefix + (rewrite_fn(body) if body.strip() else body)
        return rewrite_fn(text)

    trailing_newline = text.endswith("\n")
    chunks = split_markdown_chunks(text, max_chars=max_chunk_chars, max_words=max_chunk_words)
    if not chunks:
        return text

    out_parts: list[str] = []
    for idx, chunk in enumerate(chunks):
        if chunk.rewritable:
            rewritten_body = rewrite_fn(chunk.text) if chunk.text.strip() else chunk.text
            rendered = chunk.prefix + rewritten_body
        else:
            rendered = chunk.prefix + chunk.text

        if idx == 0:
            out_parts.append(rendered)
        else:
            out_parts.append(chunk.join_with + rendered)

    result = "".join(out_parts)
    if trailing_newline and not result.endswith("\n"):
        result += "\n"
    return result
