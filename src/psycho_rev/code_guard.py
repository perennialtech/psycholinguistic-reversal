"""Protect inline code spans, URLs, and file paths during text rewriting."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PROTECTED_PATTERN = re.compile(
    r"(`[^`\n]+`|https?://\S+|\b[a-zA-Z_][a-zA-Z0-9_]*\(\)|\b[a-zA-Z0-9_]+/[a-zA-Z0-9_./-]+\b)"
)


@dataclass
class GuardedText:
    masked: str
    placeholders: dict[str, str]

    def restore(self, rewritten: str) -> str:
        out = rewritten
        for token, original in self.placeholders.items():
            out = out.replace(token, original)
        return out


def protect_spans(text: str) -> GuardedText:
    """Replace code spans and identifiers with stable alphabetic placeholders."""
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
    return _PROTECTED_PATTERN.findall(text)


def code_preservation_rate(original: str, rewritten: str) -> float:
    """Fraction of code spans from original preserved verbatim in rewritten."""
    tokens = extract_protected_tokens(original)
    if not tokens:
        return 1.0
    kept = sum(1 for tok in tokens if tok in rewritten)
    return kept / len(tokens)
