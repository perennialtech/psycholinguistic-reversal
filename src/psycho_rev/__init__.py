"""Classical and neural reversal of psycholinguistic difficulty features in LLM text."""

from __future__ import annotations

from typing import Any

__version__ = "0.2.0"
__all__ = ["PsycholinguisticAnalyzer", "RewriteResult", "Rewriter", "__version__"]


def __getattr__(name: str) -> Any:
    if name in ("Rewriter", "RewriteResult"):
        from psycho_rev import rewriter

        return getattr(rewriter, name)
    if name == "PsycholinguisticAnalyzer":
        from psycho_rev.features import PsycholinguisticAnalyzer

        return PsycholinguisticAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
