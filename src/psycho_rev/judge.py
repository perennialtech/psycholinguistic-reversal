"""Blind position-swapped pairwise LLM-as-a-judge evaluator using httpx."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
CRITERIA: tuple[str, ...] = ("clarity", "naturalness", "meaning_preservation")
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

_SYSTEM_PROMPT = (
    "You are an impartial expert linguistics evaluator comparing two candidate versions "
    "of an original passage. Judge strictly on readability, naturalness, and faithful "
    "preservation of meaning (including any code, identifiers, or technical facts). "
    "Do not reward verbosity or penalize concise prose. Respond with a single JSON object only."
)

_USER_PROMPT_TEMPLATE = """Compare Candidate 1 and Candidate 2 against the Original passage.

Original:
{original}

Candidate 1:
{cand_1}

Candidate 2:
{cand_2}

Score each candidate on a 1 to 5 scale (1 = poor, 5 = excellent) for:
- clarity: ease of parsing and reading without garden-path ambiguity or stacked modifiers
- naturalness: grammaticality and idiomatic human flow (penalize awkward or mangled syntax)
- meaning_preservation: factual, logical, and technical fidelity to the Original

Then pick the overall winner: "1", "2", or "tie".

Return ONLY valid JSON matching this schema:
{{
  "candidate_1": {{"clarity": 1, "naturalness": 1, "meaning_preservation": 1}},
  "candidate_2": {{"clarity": 1, "naturalness": 1, "meaning_preservation": 1}},
  "winner": "1" | "2" | "tie",
  "rationale": "brief 1-2 sentence explanation"
}}"""


@dataclass
class JudgeResult:
    """Reconciled position-swapped pairwise judgment for a single sample."""

    sample_id: str
    winner: str  # "a", "b", or "tie"
    verdict_ab: str  # "a", "b", or "tie"
    verdict_ba: str  # "a", "b", or "tie"
    agreed: bool
    scores_a: dict[str, float]
    scores_b: dict[str, float]
    rationale_ab: str = ""
    rationale_ba: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairwiseSummary:
    """Aggregated pairwise LLM-as-a-judge statistics across a set of samples."""

    method_a: str
    method_b: str
    total_pairs: int
    wins_a: int
    wins_b: int
    ties: int
    win_rate_a: float
    win_rate_b: float
    tie_rate: float
    agreement_rate: float
    mean_scores_a: dict[str, float]
    mean_scores_b: dict[str, float]
    results: list[JudgeResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "total_pairs": self.total_pairs,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "ties": self.ties,
            "win_rate_a": self.win_rate_a,
            "win_rate_b": self.win_rate_b,
            "tie_rate": self.tie_rate,
            "agreement_rate": self.agreement_rate,
            "mean_scores_a": dict(self.mean_scores_a),
            "mean_scores_b": dict(self.mean_scores_b),
            "results": [r.to_dict() for r in self.results],
        }


def _clamp_score(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid criterion score: {value!r}") from exc
    return round(min(max(num, 1.0), 5.0), 4)


def _normalize_position_winner(raw: Any) -> str:
    """Normalize raw judge winner token to '1', '2', or 'tie'."""
    tok = str(raw).strip().lower()
    if tok in {"1", "candidate_1", "candidate 1", "a", "cand_a", "first"}:
        return "1"
    if tok in {"2", "candidate_2", "candidate 2", "b", "cand_b", "second"}:
        return "2"
    if tok in {"tie", "draw", "equal", "neither", "both"}:
        return "tie"
    raise ValueError(f"Unrecognized judge winner value: {raw!r}")


def parse_judge_response(content: str) -> dict[str, Any]:
    """Parse structured JSON judgment from an LLM response string."""
    text = content.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    elif not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Judge response JSON must be an object.")

    c1_raw = data.get("candidate_1") or data.get("a") or data.get("candidate_a")
    c2_raw = data.get("candidate_2") or data.get("b") or data.get("candidate_b")
    if not isinstance(c1_raw, dict) or not isinstance(c2_raw, dict):
        raise ValueError("Judge response missing candidate_1 / candidate_2 score objects.")

    c1 = {k: _clamp_score(c1_raw[k]) for k in CRITERIA}
    c2 = {k: _clamp_score(c2_raw[k]) for k in CRITERIA}
    winner = _normalize_position_winner(data.get("winner", "tie"))
    rationale = str(data.get("rationale", "")).strip()

    return {
        "candidate_1": c1,
        "candidate_2": c2,
        "winner": winner,
        "rationale": rationale,
    }


def reconcile_swapped_verdicts(
    run_ab: dict[str, Any],
    run_ba: dict[str, Any],
    sample_id: str = "",
) -> JudgeResult:
    """Reconcile AB-order and BA-order judgments into a position-debiased verdict.

    In ``run_ab``, Candidate 1 is A and Candidate 2 is B.
    In ``run_ba``, Candidate 1 is B and Candidate 2 is A.
    If the two orderings disagree on the mapped winner, the final verdict is strictly ``"tie"``.
    """
    ab_map = {"1": "a", "2": "b", "tie": "tie"}
    ba_map = {"1": "b", "2": "a", "tie": "tie"}

    verdict_ab = ab_map[_normalize_position_winner(run_ab["winner"])]
    verdict_ba = ba_map[_normalize_position_winner(run_ba["winner"])]
    agreed = verdict_ab == verdict_ba
    final_winner = verdict_ab if agreed else "tie"

    scores_a = {
        k: round((float(run_ab["candidate_1"][k]) + float(run_ba["candidate_2"][k])) / 2.0, 4)
        for k in CRITERIA
    }
    scores_b = {
        k: round((float(run_ab["candidate_2"][k]) + float(run_ba["candidate_1"][k])) / 2.0, 4)
        for k in CRITERIA
    }

    return JudgeResult(
        sample_id=sample_id,
        winner=final_winner,
        verdict_ab=verdict_ab,
        verdict_ba=verdict_ba,
        agreed=agreed,
        scores_a=scores_a,
        scores_b=scores_b,
        rationale_ab=str(run_ab.get("rationale", "")),
        rationale_ba=str(run_ba.get("rationale", "")),
    )


def summarize_pairwise(
    results: list[JudgeResult],
    method_a: str = "A",
    method_b: str = "B",
) -> PairwiseSummary:
    """Aggregate individual JudgeResult records into a PairwiseSummary."""
    n = len(results)
    if n == 0:
        zeros = {k: 0.0 for k in CRITERIA}
        return PairwiseSummary(
            method_a=method_a,
            method_b=method_b,
            total_pairs=0,
            wins_a=0,
            wins_b=0,
            ties=0,
            win_rate_a=0.0,
            win_rate_b=0.0,
            tie_rate=0.0,
            agreement_rate=0.0,
            mean_scores_a=dict(zeros),
            mean_scores_b=dict(zeros),
            results=[],
        )

    wins_a = sum(1 for r in results if r.winner == "a")
    wins_b = sum(1 for r in results if r.winner == "b")
    ties = sum(1 for r in results if r.winner == "tie")
    agreed_count = sum(1 for r in results if r.agreed)

    mean_a = {
        k: round(sum(r.scores_a[k] for r in results) / n, 4)
        for k in CRITERIA
    }
    mean_b = {
        k: round(sum(r.scores_b[k] for r in results) / n, 4)
        for k in CRITERIA
    }

    return PairwiseSummary(
        method_a=method_a,
        method_b=method_b,
        total_pairs=n,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        win_rate_a=round(wins_a / n, 4),
        win_rate_b=round(wins_b / n, 4),
        tie_rate=round(ties / n, 4),
        agreement_rate=round(agreed_count / n, 4),
        mean_scores_a=mean_a,
        mean_scores_b=mean_b,
        results=list(results),
    )


class BlindPairwiseJudge:
    """Blind, position-swapped pairwise LLM judge over OpenRouter or OpenAI via httpx."""

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 45.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        max_concurrency: int = 4,
        temperature: float = 0.0,
        client: httpx.Client | None = None,
    ):
        resolved_provider, resolved_key, resolved_base_url, resolved_model = self._resolve_endpoint(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        self.provider = resolved_provider
        self.api_key = resolved_key
        self.base_url = resolved_base_url.rstrip("/")
        self.model = resolved_model
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = max(0.0, float(backoff_base))
        self.max_concurrency = max(1, int(max_concurrency))
        self.temperature = float(temperature)
        self._semaphore = threading.Semaphore(self.max_concurrency)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            headers=self._build_headers(),
        )

    @staticmethod
    def _resolve_endpoint(
        *,
        provider: str | None,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
    ) -> tuple[str, str, str, str]:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

        if provider is not None:
            norm_provider = provider.strip().lower()
            if norm_provider not in {"openrouter", "openai"}:
                raise ValueError(f"Unsupported provider {provider!r}; expected 'openrouter' or 'openai'.")
        elif base_url is not None and "openai.com" in base_url:
            norm_provider = "openai"
        elif base_url is not None and "openrouter.ai" in base_url:
            norm_provider = "openrouter"
        elif openrouter_key:
            norm_provider = "openrouter"
        elif openai_key:
            norm_provider = "openai"
        else:
            norm_provider = "openrouter"

        if api_key is not None:
            resolved_key = api_key.strip()
        elif norm_provider == "openrouter":
            resolved_key = openrouter_key
        else:
            resolved_key = openai_key

        if not resolved_key:
            raise ValueError(
                "Missing API key for BlindPairwiseJudge: set OPENROUTER_API_KEY or OPENAI_API_KEY, "
                "or pass api_key explicitly."
            )

        if base_url is not None:
            resolved_base_url = base_url
        elif norm_provider == "openrouter":
            resolved_base_url = OPENROUTER_BASE_URL
        else:
            resolved_base_url = OPENAI_BASE_URL

        if model is not None:
            resolved_model = model
        elif norm_provider == "openrouter":
            resolved_model = DEFAULT_OPENROUTER_MODEL
        else:
            resolved_model = DEFAULT_OPENAI_MODEL

        return norm_provider, resolved_key, resolved_base_url, resolved_model

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["X-Title"] = "psycholinguistic-reversal-eval"
        return headers

    def _chat_completion_url(self) -> str:
        if self._owns_client:
            return "/chat/completions"
        return f"{self.base_url}/chat/completions"

    def _call_single_order(
        self,
        original: str,
        cand_1: str,
        cand_2: str,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_PROMPT_TEMPLATE.format(
                        original=original,
                        cand_1=cand_1,
                        cand_2=cand_2,
                    ),
                },
            ],
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._semaphore:
                    resp = self._client.post(
                        self._chat_completion_url(),
                        json=payload,
                        headers=self._build_headers(),
                        timeout=self.timeout,
                    )
                if resp.status_code in _RETRYABLE_STATUS_CODES:
                    raise httpx.HTTPStatusError(
                        f"Retryable HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                body = resp.json()
                content = body["choices"][0]["message"]["content"]
                return parse_judge_response(content)
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, ValueError, KeyError, json.JSONDecodeError) as exc:
                last_exc = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in _RETRYABLE_STATUS_CODES:
                    raise
                if attempt >= self.max_retries:
                    break
                delay = self.backoff_base * (2**attempt)
                if delay > 0:
                    time.sleep(delay)

        assert last_exc is not None
        raise RuntimeError(f"BlindPairwiseJudge failed after {self.max_retries + 1} attempts: {last_exc}") from last_exc

    def judge_pair(
        self,
        original: str,
        cand_a: str,
        cand_b: str,
        *,
        sample_id: str = "",
    ) -> JudgeResult:
        """Evaluate ``(cand_a, cand_b)`` in both AB and BA orders and reconcile strictly."""
        run_ab = self._call_single_order(original, cand_a, cand_b)
        run_ba = self._call_single_order(original, cand_b, cand_a)
        return reconcile_swapped_verdicts(run_ab, run_ba, sample_id=sample_id)

    def evaluate_pairs(
        self,
        pairs: list[tuple[str, str, str, str]],
        *,
        method_a: str = "A",
        method_b: str = "B",
    ) -> PairwiseSummary:
        """Evaluate a list of ``(sample_id, original, cand_a, cand_b)`` tuples concurrently."""
        if not pairs:
            return summarize_pairwise([], method_a=method_a, method_b=method_b)

        ordered_results: list[JudgeResult | None] = [None] * len(pairs)
        if self.max_concurrency == 1 or len(pairs) == 1:
            for idx, (sample_id, orig, cand_a, cand_b) in enumerate(pairs):
                ordered_results[idx] = self.judge_pair(orig, cand_a, cand_b, sample_id=sample_id)
        else:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                future_to_idx = {
                    pool.submit(
                        self.judge_pair, orig, cand_a, cand_b, sample_id=sample_id
                    ): idx
                    for idx, (sample_id, orig, cand_a, cand_b) in enumerate(pairs)
                }
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    ordered_results[idx] = fut.result()

        finalized = [r for r in ordered_results if r is not None]
        return summarize_pairwise(finalized, method_a=method_a, method_b=method_b)

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    def __enter__(self) -> "BlindPairwiseJudge":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
