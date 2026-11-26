"""High-level lazy-loading Python SDK for psycholinguistic text reversal."""

from __future__ import annotations

import difflib
import inspect
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from psycho_rev.code_guard import code_preservation_rate

DEFAULT_LORA_ADAPTER = "8q0sb/qwen25-3b-psycholinguistic-reversal-lora"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SUPPORTED_BACKENDS: tuple[str, ...] = (
    "classical",
    "openai",
    "openrouter",
    "lora",
    "best-of-n",
    "hybrid-bon",
    "hybrid-lora",
    "zeroshot",
    "rtt",
)

_BACKEND_ALIASES: dict[str, str] = {
    "qwen": "zeroshot",
    "nllb": "rtt",
}


@dataclass
class RewriteResult:
    """Structured output returned by :class:`Rewriter`."""

    original: str
    rewritten: str
    changed: bool
    skipped_as_clear: bool
    backend: str
    latency_ms: float
    code_preservation: float
    before_difficulty: float | None = None
    after_difficulty: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unified_diff(original: str, rewritten: str, filename: str = "input") -> str:
    """Return a unified diff between *original* and *rewritten* text.

    Returns an empty string when *original* and *rewritten* are identical.
    """
    if original == rewritten:
        return ""

    orig_lines = original.splitlines(keepends=True)
    new_lines = rewritten.splitlines(keepends=True)
    if orig_lines and not orig_lines[-1].endswith("\n"):
        orig_lines[-1] += "\n"
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    diff_iter = difflib.unified_diff(
        orig_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    return "".join(diff_iter)


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only keyword arguments accepted by *callable_obj*."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {k: v for k, v in kwargs.items() if v is not None}

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {k: v for k, v in kwargs.items() if v is not None}

    return {k: v for k, v in kwargs.items() if k in params and v is not None}


def _normalize_backend(name: str) -> str:
    key = name.strip().lower()
    key = _BACKEND_ALIASES.get(key, key)
    if key not in SUPPORTED_BACKENDS:
        valid = ", ".join(SUPPORTED_BACKENDS)
        raise ValueError(f"Unsupported backend {name!r}. Choose one of: {valid}")
    return key


class Rewriter:
    """Unified, lazy-loading interface for psycholinguistic analysis and rewriting."""

    def __init__(
        self,
        backend: str = "classical",
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        adapter_dir: str | Path | None = None,
        samples: int = 6,
        skip_if_clear: bool = True,
        safe_mode: bool = True,
        compute_metrics: bool = True,
        analyzer: Any | None = None,
        reverser: Any | None = None,
    ) -> None:
        self.backend = _normalize_backend(backend)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.adapter_dir = str(adapter_dir) if adapter_dir is not None else DEFAULT_LORA_ADAPTER
        self.samples = samples
        self.skip_if_clear = skip_if_clear
        self.safe_mode = safe_mode
        self.compute_metrics = compute_metrics

        self._analyzer: Any | None = analyzer
        self._reverser: Any | None = reverser
        self._owned_resources: list[Any] = []

    def _get_analyzer(self) -> Any:
        if self._analyzer is None:
            from psycho_rev.features import PsycholinguisticAnalyzer

            self._analyzer = PsycholinguisticAnalyzer()
        return self._analyzer

    def _resolve_api_credentials(self) -> tuple[str | None, str]:
        if self.backend == "openrouter":
            key = self.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            url = self.base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL
            return key, url
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        url = self.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        return key, url

    def _get_reverser(self) -> Any:
        if self._reverser is not None:
            return self._reverser

        if self.backend == "classical":
            from psycho_rev.classical import ClassicalReverser

            kwargs = _filter_kwargs(ClassicalReverser.__init__, {"safe_mode": self.safe_mode})
            self._reverser = ClassicalReverser(**kwargs)
            return self._reverser

        if self.backend in ("openai", "openrouter"):
            from psycho_rev import neural

            api_cls = getattr(neural, "OpenAICompatibleReverser", None)
            if api_cls is None:
                raise RuntimeError(
                    "OpenAICompatibleReverser is not available in psycho_rev.neural."
                )
            api_key, base_url = self._resolve_api_credentials()
            if not api_key:
                env_var = "OPENROUTER_API_KEY" if self.backend == "openrouter" else "OPENAI_API_KEY"
                raise ValueError(
                    f"Missing API key for {self.backend!r} backend. "
                    f"Pass api_key=... or set {env_var}."
                )
            candidate_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "base_url": base_url,
                "model": self.model,
                "skip_if_clear": self.skip_if_clear,
                "analyzer": self._analyzer,
            }
            kwargs = _filter_kwargs(api_cls.__init__, candidate_kwargs)
            self._reverser = api_cls(**kwargs)
            self._owned_resources.append(self._reverser)
            return self._reverser

        if self.backend == "rtt":
            from psycho_rev.neural import NLLBRoundTripReverser

            kwargs = _filter_kwargs(NLLBRoundTripReverser.__init__, {"model_name": self.model})
            self._reverser = NLLBRoundTripReverser(**kwargs)
            self._owned_resources.append(self._reverser)
            return self._reverser

        if self.backend == "zeroshot":
            from psycho_rev.neural import QwenZeroShotReverser

            kwargs = _filter_kwargs(
                QwenZeroShotReverser.__init__,
                {
                    "model_name": self.model or DEFAULT_BASE_MODEL,
                    "skip_if_clear": self.skip_if_clear,
                },
            )
            self._reverser = QwenZeroShotReverser(**kwargs)
            self._owned_resources.append(self._reverser)
            return self._reverser

        if self.backend in ("best-of-n", "hybrid-bon"):
            from psycho_rev.neural import (
                BestOfNPsycholinguisticReverser,
                HybridReverser,
                QwenZeroShotReverser,
            )
            from psycho_rev.semantics import SemanticEvaluator

            analyzer = self._get_analyzer()
            gen_kwargs = _filter_kwargs(
                QwenZeroShotReverser.__init__,
                {"model_name": self.model or DEFAULT_BASE_MODEL},
            )
            gen = QwenZeroShotReverser(**gen_kwargs)
            self._owned_resources.append(gen)
            sem = SemanticEvaluator()
            bon_kwargs = _filter_kwargs(
                BestOfNPsycholinguisticReverser.__init__,
                {
                    "num_samples": self.samples,
                    "skip_if_clear": self.skip_if_clear,
                },
            )
            bon = BestOfNPsycholinguisticReverser(gen, analyzer, sem, **bon_kwargs)
            if self.backend == "hybrid-bon":
                classical_kwargs = _filter_kwargs(
                    HybridReverser.__init__,
                    {"safe_mode": self.safe_mode, "skip_if_clear": self.skip_if_clear},
                )
                self._reverser = HybridReverser(bon, **classical_kwargs)
            else:
                self._reverser = bon
            return self._reverser

        if self.backend in ("lora", "hybrid-lora"):
            from psycho_rev.neural import HybridReverser, LoRASinglePassReverser

            lora_kwargs = _filter_kwargs(
                LoRASinglePassReverser.__init__,
                {
                    "base_model": self.model,
                    "skip_if_clear": self.skip_if_clear,
                },
            )
            lora = LoRASinglePassReverser(self.adapter_dir, **lora_kwargs)
            self._owned_resources.append(lora)
            if self.backend == "hybrid-lora":
                hybrid_kwargs = _filter_kwargs(
                    HybridReverser.__init__,
                    {"safe_mode": self.safe_mode, "skip_if_clear": self.skip_if_clear},
                )
                self._reverser = HybridReverser(lora, **hybrid_kwargs)
            else:
                self._reverser = lora
            return self._reverser

        raise ValueError(f"Unsupported backend: {self.backend!r}")

    def analyze(self, text: str) -> Any:
        """Compute the full psycholinguistic :class:`FeatureReport` for *text*."""
        return self._get_analyzer().analyze(text)

    def _call_reverser(self, reverser: Any, text: str, skip_if_clear: bool) -> tuple[str, bool]:
        rewrite_fn = getattr(reverser, "rewrite", None)
        if rewrite_fn is None and callable(reverser):
            rewrite_fn = reverser
        if rewrite_fn is None:
            raise TypeError(f"Backend reverser {reverser!r} has no callable rewrite method.")

        call_kwargs = _filter_kwargs(rewrite_fn, {"skip_if_clear": skip_if_clear})
        raw = rewrite_fn(text, **call_kwargs)

        skipped = False
        if isinstance(raw, str):
            rewritten = raw
            skipped = bool(getattr(reverser, "last_skipped_as_clear", False))
        elif hasattr(raw, "rewritten"):
            rewritten = str(raw.rewritten)
            skipped = bool(getattr(raw, "skipped_as_clear", False))
        else:
            rewritten = str(raw)

        return rewritten, skipped

    def rewrite(
        self,
        text: str,
        *,
        skip_if_clear: bool | None = None,
        compute_metrics: bool | None = None,
    ) -> RewriteResult:
        """Rewrite *text* and return a structured :class:`RewriteResult`."""
        effective_skip = self.skip_if_clear if skip_if_clear is None else skip_if_clear
        effective_metrics = self.compute_metrics if compute_metrics is None else compute_metrics

        t0 = time.perf_counter()

        before_difficulty: float | None = None
        if effective_metrics:
            before_report = self.analyze(text)
            before_difficulty = float(before_report.difficulty_score)

        if not text or not text.strip():
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return RewriteResult(
                original=text,
                rewritten=text,
                changed=False,
                skipped_as_clear=True,
                backend=self.backend,
                latency_ms=round(elapsed_ms, 2),
                code_preservation=1.0,
                before_difficulty=before_difficulty,
                after_difficulty=before_difficulty,
            )

        reverser = self._get_reverser()
        rewritten, skipped = self._call_reverser(reverser, text, effective_skip)
        changed = rewritten != text

        after_difficulty: float | None = None
        if effective_metrics:
            if not changed:
                after_difficulty = before_difficulty
            else:
                after_report = self.analyze(rewritten)
                after_difficulty = float(after_report.difficulty_score)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        preservation = round(code_preservation_rate(text, rewritten), 4)

        return RewriteResult(
            original=text,
            rewritten=rewritten,
            changed=changed,
            skipped_as_clear=skipped,
            backend=self.backend,
            latency_ms=round(elapsed_ms, 2),
            code_preservation=preservation,
            before_difficulty=before_difficulty,
            after_difficulty=after_difficulty,
        )

    def rewrite_batch(
        self,
        texts: list[str],
        *,
        skip_if_clear: bool | None = None,
        compute_metrics: bool | None = None,
    ) -> list[RewriteResult]:
        """Rewrite a sequence of passages and return a list of :class:`RewriteResult`."""
        return [
            self.rewrite(
                item,
                skip_if_clear=skip_if_clear,
                compute_metrics=compute_metrics,
            )
            for item in texts
        ]

    def rewrite_file(
        self,
        path: str | Path,
        *,
        in_place: bool = False,
        encoding: str = "utf-8",
        skip_if_clear: bool | None = None,
        compute_metrics: bool | None = None,
    ) -> RewriteResult:
        """Read a UTF-8 file, rewrite its contents, and optionally save changes in place."""
        file_path = Path(path)
        original = file_path.read_text(encoding=encoding)
        result = self.rewrite(
            original,
            skip_if_clear=skip_if_clear,
            compute_metrics=compute_metrics,
        )
        if in_place and result.changed:
            file_path.write_text(result.rewritten, encoding=encoding)
        return result

    def close(self) -> None:
        """Release any underlying model or HTTP client resources."""
        for resource in self._owned_resources:
            close_fn = getattr(resource, "close", None)
            if callable(close_fn):
                close_fn()
        self._owned_resources.clear()

    def __enter__(self) -> "Rewriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
