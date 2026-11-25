"""Protect code blocks, inline code spans, URLs, identifiers, and paths during text rewriting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

_CODE_EXTENSIONS = (
    "py|pyi|pyx|js|mjs|cjs|ts|tsx|jsx|rs|go|c|h|cc|cpp|cxx|hpp|hh|"
    "java|kt|kts|scala|rb|php|pl|sh|bash|zsh|fish|ps1|bat|cmd|cs|fs|"
    "swift|dart|lua|r|R|jl|ex|exs|erl|hs|ml|zig|nim|sql|html|htm|css|"
    "scss|sass|less|xml|json|jsonc|jsonl|yaml|yml|toml|ini|cfg|conf|"
    "env|lock|md|rst|tex|csv|tsv|proto|graphql|gql|tf|hcl|nix|mk|"
    "cmake|gradle|txt|log"
)

_PROSE_SLASH_NEG = (
    r"(?!(?:and|or|yes|no|on|off|he|she|his|her|him|either|neither|"
    r"true|false|in|out|up|down|input|output|read|write|client|server|"
    r"before|after|pros|cons)/)"
)

_FENCED_BLOCK_RE = re.compile(r"(?m)^[ \t]*(?:```|~~~)[^\n]*\n[\s\S]*?^[ \t]*(?:```|~~~)[ \t]*$")

_PROTECTED_PATTERN = re.compile(
    r"("
    # 1. Fenced code blocks (triple backticks or tildes)
    r"```[\s\S]*?```"
    r"|~~~[\s\S]*?~~~"
    # 2. Inline backticks
    r"|`[^`\n]+`"
    # 3. URLs (excluding trailing sentence punctuation)
    r"|https?://[^\s<>()\[\]]+(?<![.,;:!?])"
    # 4. Dotted module/method calls with parentheses (e.g. Pool.acquire(), os.path.join())
    r"|\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\(\)"
    # 5. Bare function calls (e.g. parse_header(), is_file())
    r"|\b[A-Za-z_][A-Za-z0-9_]*\(\)"
    # 6. Slash paths (e.g. src/cache/store.py, ./run.sh, /etc/hosts), excluding common prose pairs
    r"|(?<![A-Za-z0-9_./-])(?:(?:\.{1,2}/|/)[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)"
    r"|\b" + _PROSE_SLASH_NEG + r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+\b"
    # 7. Bare filenames with common code/config extensions (e.g. store.py, routes.yaml, Cargo.toml)
    r"|(?<![A-Za-z0-9_./-])\.?[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:" + _CODE_EXTENSIONS + r")\b"
    # 8. Dotted attributes / module paths without parens (segments >= 2 chars so e.g./i.e./U.S. stay prose)
    r"|\b[A-Za-z_][A-Za-z0-9_]+(?:\.[A-Za-z_][A-Za-z0-9_]+)+\b(?!\.)"
    r")"
)

_FENCE_OPEN_RE = re.compile(r"^([ \t]*)(`{3,}|~{3,})(.*)$")
_HEADING_RE = re.compile(r"^([ \t]*#{1,6}[ \t]+)(.*)$")
_LIST_ITEM_RE = re.compile(r"^([ \t]*(?:[-*+]|\d+[.)])[ \t]+)(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^([ \t]*>[ \t]*)(.*)$")


@dataclass
class GuardedText:
    masked: str
    placeholders: dict[str, str]

    def restore(self, rewritten: str) -> str:
        out = rewritten
        # Restore longer placeholder keys first (e.g. CODEITEM10 before CODEITEM1)
        for token in sorted(self.placeholders, key=len, reverse=True):
            out = out.replace(token, self.placeholders[token])
        return out


def strip_fenced_code_blocks(text: str) -> str:
    """Remove fenced code blocks so downstream prose analyzers do not count code as prose."""
    return re.sub(r"```[\s\S]*?```|~~~[\s\S]*?~~~", "", text)


def protect_spans(text: str) -> GuardedText:
    """Replace code spans, fenced blocks, URLs, paths, and identifiers with alphabetic placeholders."""
    placeholders: dict[str, str] = {}
    counter = 0

    def _repl(match: re.Match[str]) -> str:
        nonlocal counter
        original = match.group(0)
        key = f"CODEITEM{counter}"
        counter += 1
        placeholders[key] = original
        return key

    masked = _PROTECTED_PATTERN.sub(_repl, text)
    return GuardedText(masked=masked, placeholders=placeholders)


def extract_protected_tokens(text: str) -> list[str]:
    """Return all code-like spans in a passage for fidelity verification."""
    return [m.group(0) for m in _PROTECTED_PATTERN.finditer(text)]


def code_preservation_rate(original: str, rewritten: str) -> float:
    """Fraction of code spans from original preserved verbatim in rewritten."""
    tokens = extract_protected_tokens(original)
    if not tokens:
        return 1.0
    kept = sum(1 for tok in tokens if tok in rewritten)
    return kept / len(tokens)


def rewrite_markdown_preserving_structure(text: str, rewrite_fn: Callable[[str], str]) -> str:
    """Rewrite prose inside a markdown or plain-text document while preserving structure.

    Fenced code blocks and blank lines are preserved verbatim. Leading markdown
    heading markers (`# `), list prefixes (`- `, `* `, `1. `), and blockquote
    prefixes (`> `) are preserved while their prose bodies are passed to `rewrite_fn`.
    Single-line plain text without markdown markers behaves identically to `rewrite_fn(text)`.
    """
    if not text:
        return text

    if "\n" not in text:
        if _FENCE_OPEN_RE.match(text):
            return text
        for pattern in (_HEADING_RE, _LIST_ITEM_RE, _BLOCKQUOTE_RE):
            m = pattern.match(text)
            if m:
                prefix, body = m.group(1), m.group(2)
                return prefix + (rewrite_fn(body) if body.strip() else body)
        return rewrite_fn(text)

    trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    out_lines: list[str] = []
    prose_buf: list[str] = []

    def _flush_prose() -> None:
        if not prose_buf:
            return
        paragraph = " ".join(part.strip() for part in prose_buf)
        out_lines.append(rewrite_fn(paragraph) if paragraph.strip() else paragraph)
        prose_buf.clear()

    in_fence = False
    fence_char = ""
    fence_len = 0

    for line in lines:
        if in_fence:
            out_lines.append(line)
            stripped = line.strip()
            if stripped.startswith(fence_char * fence_len) and set(stripped) == {fence_char}:
                in_fence = False
                fence_char = ""
                fence_len = 0
            continue

        fence_match = _FENCE_OPEN_RE.match(line)
        if fence_match:
            _flush_prose()
            out_lines.append(line)
            in_fence = True
            fence_marker = fence_match.group(2)
            fence_char = fence_marker[0]
            fence_len = len(fence_marker)
            continue

        if not line.strip():
            _flush_prose()
            out_lines.append(line)
            continue

        matched_prefix = False
        for pattern in (_HEADING_RE, _LIST_ITEM_RE, _BLOCKQUOTE_RE):
            m = pattern.match(line)
            if m:
                _flush_prose()
                prefix, body = m.group(1), m.group(2)
                out_lines.append(prefix + (rewrite_fn(body) if body.strip() else body))
                matched_prefix = True
                break
        if matched_prefix:
            continue

        prose_buf.append(line)

    _flush_prose()
    result = "\n".join(out_lines)
    if trailing_newline:
        result += "\n"
    return result
