"""Neural and API-backed psycholinguistic reversal pipelines.

Implements six inference pipelines:
1. OpenAICompatibleReverser: Production httpx client for OpenAI, OpenRouter, vLLM, Ollama, and llama.cpp servers (zero local PyTorch/GPU overhead)
2. NLLBRoundTripReverser: English -> Japanese -> English bilingual pivoting (NLLB-200)
3. QwenZeroShotReverser: Single-pass local open-weight instruct LLM rewriter
4. BestOfNPsycholinguisticReverser: Sample-and-rerank with acceptability penalty, NLI floor, and hard code preservation
5. LoRASinglePassReverser: Single-pass local open-weight LLM + fine-tuned LoRA adapter
6. HybridReverser: Any neural/API reverser followed by safe-mode ClassicalReverser post-processing
"""

from __future__ import annotations

import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import httpx

from psycho_rev.classical import ClassicalReverser
from psycho_rev.code_guard import (
    code_preservation_rate,
    protect_spans,
    rewrite_markdown_preserving_structure,
)

if TYPE_CHECKING:
    from psycho_rev.features import PsycholinguisticAnalyzer
    from psycho_rev.semantics import SemanticEvaluator

REWRITE_SYSTEM_PROMPT = (
    "You are a careful copy editor who makes dense, convoluted English clear, direct, and natural "
    "to read while preserving the exact meaning, tone, and every factual detail.\n"
    "Guidelines:\n"
    "1. Preserve every placeholder token such as CODEITEM0, CODEITEM1 verbatim without altering, "
    "moving out of context, or dropping it.\n"
    "2. Unpack long chains of stacked nouns into natural prepositional or relative phrases.\n"
    "3. Prefer concrete subjects and plain, literal verbs over figurative personification or "
    "jargon-heavy noun-to-verb coinages.\n"
    "4. Keep object relative clauses clear by including an explicit 'that' or 'which' when "
    "omitting it creates ambiguity or a garden path.\n"
    "5. Break overly long, tangled sentences into shorter, direct ones only when doing so "
    "improves readability.\n"
    "6. If a sentence is already clear and natural, leave it unchanged rather than paraphrasing "
    "for the sake of change.\n"
    "7. Output ONLY the rewritten text, with no commentary, preamble, surrounding quotes, or "
    "extra markdown."
)

_RETRYABLE_STATUS_CODES: set[int] = {429, 500, 502, 503, 504}


def _clean_output(text: str) -> str:
    out = text.strip()
    if len(out) >= 2 and out[0] == '"' and out[-1] == '"':
        out = out[1:-1].strip()
    return " ".join(out.split())


def is_already_clear(
    text: str,
    analyzer: Any,
    threshold: float = 35.0,
) -> bool:
    """Return True when prose is already below the psycholinguistic difficulty threshold."""
    if not text or not text.strip():
        return True
    report = analyzer.analyze(text)
    return float(report.difficulty_score) <= threshold


class OpenAICompatibleReverser:
    """Production HTTP rewriter for any OpenAI-compatible chat completions endpoint.

    Works with OpenAI, OpenRouter, vLLM, Ollama, and llama.cpp server using ``httpx``
    with zero local PyTorch/GPU overhead. Includes explicit timeouts, exponential-backoff
    retries on transient failures, bounded-concurrency batching, code-span preservation,
    and an optional clarity no-op gate so already-readable prose is returned untouched.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        max_concurrency: int = 4,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        skip_if_clear: bool = True,
        clear_difficulty_threshold: float = 35.0,
        analyzer: Any | None = None,
        fallback_on_code_loss: bool = True,
        max_chunk_chars: int = 1800,
        client: httpx.Client | None = None,
    ):
        self.model = model
        cleaned_url = base_url.rstrip("/")
        if cleaned_url.endswith("/chat/completions"):
            self.endpoint = cleaned_url
        else:
            self.endpoint = f"{cleaned_url}/chat/completions"
        self.base_url = cleaned_url
        self.api_key = (
            api_key
            if api_key is not None
            else (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))
        )
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self.max_concurrency = max(1, int(max_concurrency))
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.extra_headers = dict(extra_headers or {})
        self.skip_if_clear = skip_if_clear
        self.clear_difficulty_threshold = float(clear_difficulty_threshold)
        self._analyzer = analyzer
        self.fallback_on_code_loss = fallback_on_code_loss
        self.max_chunk_chars = max_chunk_chars

        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout))

    def _get_analyzer(self) -> Any:
        if self._analyzer is None:
            from psycho_rev.features import PsycholinguisticAnalyzer

            self._analyzer = PsycholinguisticAnalyzer()
        return self._analyzer

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _post_completion(
        self,
        masked_text: str,
        temperature: float,
        top_p: float,
        n: int = 1,
    ) -> list[str]:
        max_tok = (
            self.max_tokens
            if self.max_tokens is not None
            else min(max(int(len(masked_text.split()) * 2.4) + 64, 128), 2048)
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Rewrite this passage clearly:\n\n{masked_text}"},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tok,
        }
        if n > 1:
            payload["n"] = n

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=payload,
                )
                if resp.status_code in _RETRYABLE_STATUS_CODES:
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff * (2**attempt))
                        continue
                    resp.raise_for_status()
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                outputs: list[str] = []
                for choice in choices:
                    msg = choice.get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        outputs.append(content)
                return outputs
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                is_retryable_http = (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response is not None
                    and exc.response.status_code in _RETRYABLE_STATUS_CODES
                )
                if (
                    isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))
                    or is_retryable_http
                ) and attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (2**attempt))
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        return []

    def generate_candidates(
        self,
        text: str,
        num_samples: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.92,
        include_greedy: bool = True,
    ) -> list[str]:
        """Generate candidate rewrites for compatibility with BestOfNPsycholinguisticReverser."""
        guarded = protect_spans(text)
        raw_outputs: list[str] = []
        if include_greedy:
            raw_outputs.extend(
                self._post_completion(guarded.masked, temperature=0.0, top_p=1.0, n=1)
            )
        n_stoch = num_samples - (1 if include_greedy else 0)
        if n_stoch > 0:
            raw_outputs.extend(
                self._post_completion(
                    guarded.masked, temperature=temperature, top_p=top_p, n=n_stoch
                )
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for raw in raw_outputs:
            cleaned = _clean_output(guarded.restore(raw))
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)
        return deduped or [text]

    def _rewrite_chunk(self, text: str) -> str:
        if not text.strip():
            return text
        if self.skip_if_clear and is_already_clear(
            text, self._get_analyzer(), threshold=self.clear_difficulty_threshold
        ):
            return text

        guarded = protect_spans(text)
        outputs = self._post_completion(
            guarded.masked, temperature=self.temperature, top_p=self.top_p, n=1
        )
        result = _clean_output(guarded.restore(outputs[0])) if outputs else text

        if self.fallback_on_code_loss and code_preservation_rate(text, result) < 1.0:
            retry_outputs = self._post_completion(
                guarded.masked, temperature=0.0, top_p=1.0, n=1
            )
            if retry_outputs:
                retry_result = _clean_output(guarded.restore(retry_outputs[0]))
                if retry_result and code_preservation_rate(text, retry_result) >= 1.0:
                    return retry_result
            return text

        return result or text

    def rewrite(self, text: str) -> str:
        return rewrite_markdown_preserving_structure(
            text,
            self._rewrite_chunk,
            max_chunk_chars=self.max_chunk_chars,
        )

    def rewrite_batch(
        self,
        texts: list[str],
        max_concurrency: int | None = None,
    ) -> list[str]:
        """Rewrite a list of documents concurrently while preserving input order."""
        if not texts:
            return []
        workers = max(1, int(max_concurrency or self.max_concurrency))
        if workers == 1 or len(texts) == 1:
            return [self.rewrite(t) for t in texts]
        with ThreadPoolExecutor(max_workers=min(workers, len(texts))) as pool:
            return list(pool.map(self.rewrite, texts))

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    def __enter__(self) -> OpenAICompatibleReverser:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class NLLBRoundTripReverser:
    """Round-trip translation via Japanese using NLLB-200 (Clark et al. 2026 baseline)."""

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        pivot_lang: str = "jpn_Jpan",
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pivot_lang = pivot_lang
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.max_length = None
        self.model.eval()

    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        import torch

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
            self.device
        )
        tgt_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_id,
                max_new_tokens=256,
                num_beams=4,
            )
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

    def _rewrite_chunk(self, text: str) -> str:
        guarded = protect_spans(text)
        ja = self._translate(guarded.masked, "eng_Latn", self.pivot_lang)
        en = self._translate(ja, self.pivot_lang, "eng_Latn")
        return _clean_output(guarded.restore(en))

    def rewrite(self, text: str) -> str:
        return rewrite_markdown_preserving_structure(text, self._rewrite_chunk)

    def close(self) -> None:
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class QwenZeroShotReverser:
    """Single-pass instruct LLM rewriter with psycholinguistic de-compression prompt."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str | None = None,
        skip_if_clear: bool = True,
        clear_difficulty_threshold: float = 35.0,
        analyzer: Any | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_if_clear = skip_if_clear
        self.clear_difficulty_threshold = float(clear_difficulty_threshold)
        self._analyzer = analyzer
        dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        if self.device != "cuda":
            dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.max_length = None
        self.model.eval()

    def _get_analyzer(self) -> Any:
        if self._analyzer is None:
            from psycho_rev.features import PsycholinguisticAnalyzer

            self._analyzer = PsycholinguisticAnalyzer()
        return self._analyzer

    def _build_prompt(self, masked_text: str) -> str:
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Rewrite this passage clearly:\n\n{masked_text}"},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_candidates(
        self,
        text: str,
        num_samples: int = 1,
        temperature: float = 0.8,
        top_p: float = 0.92,
        include_greedy: bool = True,
    ) -> list[str]:
        import torch

        guarded = protect_spans(text)
        prompt = self._build_prompt(guarded.masked)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        max_new = min(max(int(len(guarded.masked.split()) * 2.4) + 32, 96), 384)

        results: list[str] = []
        with torch.inference_mode():
            if include_greedy:
                g_out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                decoded = self.tokenizer.decode(g_out[0][prompt_len:], skip_special_tokens=True)
                results.append(_clean_output(guarded.restore(decoded)))

            n_stoch = num_samples - (1 if include_greedy else 0)
            if n_stoch > 0:
                s_out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=n_stoch,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                for seq in s_out:
                    decoded = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    results.append(_clean_output(guarded.restore(decoded)))

        deduped: list[str] = []
        seen: set[str] = set()
        for r in results:
            if r and r not in seen:
                seen.add(r)
                deduped.append(r)
        return deduped or [text]

    def _rewrite_chunk(self, text: str) -> str:
        if self.skip_if_clear and is_already_clear(
            text, self._get_analyzer(), threshold=self.clear_difficulty_threshold
        ):
            return text
        return self.generate_candidates(text, num_samples=1, include_greedy=True)[0]

    def rewrite(self, text: str) -> str:
        return rewrite_markdown_preserving_structure(text, self._rewrite_chunk)

    def close(self) -> None:
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class RankedCandidate:
    text: str
    objective: float
    difficulty_score: float
    cosine_similarity: float
    nli_min_entailment: float
    code_preservation: float
    nli_max_contradiction: float = 0.0
    gpt2_perplexity: float = 0.0
    length_ratio: float = 1.0
    acceptability_min: float = 1.0


class BestOfNPsycholinguisticReverser:
    """Inference-time search: sample K rewrites and select the best readability + fidelity trade-off.

    Decouples from raw difficulty-score gaming by capping difficulty reduction reward,
    penalizing grammatical acceptability drops, enforcing hard code preservation and an
    NLI entailment floor, and preferring concise, natural rewrites.
    """

    def __init__(
        self,
        generator: Any,
        analyzer: PsycholinguisticAnalyzer,
        semantics: SemanticEvaluator,
        num_samples: int = 6,
        classical_fallback: ClassicalReverser | None = None,
        nli_floor: float = 0.65,
        max_contradiction: float = 0.25,
        skip_if_clear: bool = True,
        clear_difficulty_threshold: float = 35.0,
    ):
        self.generator = generator
        self.analyzer = analyzer
        self.semantics = semantics
        self.num_samples = num_samples
        self._classical_fallback = classical_fallback
        self.nli_floor = float(nli_floor)
        self.max_contradiction = float(max_contradiction)
        self.skip_if_clear = skip_if_clear
        self.clear_difficulty_threshold = float(clear_difficulty_threshold)

    def _get_classical_fallback(self) -> ClassicalReverser:
        if self._classical_fallback is None:
            self._classical_fallback = ClassicalReverser(safe_mode=True)
        return self._classical_fallback

    def _passes_hard_constraints(self, cand: RankedCandidate) -> bool:
        return (
            cand.code_preservation >= 1.0
            and cand.nli_min_entailment >= self.nli_floor
            and cand.nli_max_contradiction <= self.max_contradiction
        )

    def _score_candidates(self, text: str, candidates: list[str]) -> list[RankedCandidate]:
        orig_feat = self.analyzer.analyze(text)
        sem_reports = self.semantics.evaluate_batch([text] * len(candidates), candidates)
        scored: list[RankedCandidate] = []
        for cand, sem in zip(candidates, sem_reports):
            feat = self.analyzer.analyze(cand)
            # Cap difficulty reduction reward to prevent metric-gaming via degenerate simplification
            diff_gain = min(max(orig_feat.difficulty_score - feat.difficulty_score, -25.0), 30.0)
            len_pen = (
                max(0.0, sem.length_ratio - 1.45) * 18.0
                + max(0.0, 0.78 - sem.length_ratio) * 22.0
            )
            code_pen = (1.0 - sem.code_preservation) * 120.0
            nli_contra = float(getattr(sem, "nli_max_contradiction", 0.0))
            nli_floor_pen = max(0.0, self.nli_floor - sem.nli_min_entailment) * 80.0
            acc_min = float(getattr(sem, "acceptability_min", 1.0))
            acc_pen = max(0.0, 0.85 - acc_min) * 60.0 + max(0.0, 1.0 - acc_min) * 20.0
            ppl_pen = 0.08 * min(float(sem.gpt2_perplexity), 200.0)
            obj = (
                diff_gain
                + 40.0 * sem.cosine_similarity
                + 50.0 * sem.nli_min_entailment
                - 60.0 * nli_contra
                - nli_floor_pen
                - acc_pen
                - ppl_pen
                - len_pen
                - code_pen
            )
            scored.append(
                RankedCandidate(
                    text=cand,
                    objective=round(obj, 3),
                    difficulty_score=feat.difficulty_score,
                    cosine_similarity=sem.cosine_similarity,
                    nli_min_entailment=sem.nli_min_entailment,
                    code_preservation=sem.code_preservation,
                    nli_max_contradiction=round(nli_contra, 4),
                    gpt2_perplexity=sem.gpt2_perplexity,
                    length_ratio=sem.length_ratio,
                    acceptability_min=round(acc_min, 4),
                )
            )
        scored.sort(key=lambda c: c.objective, reverse=True)
        return scored

    def rewrite_ranked_all(self, text: str) -> list[RankedCandidate]:
        """Generate, score, and rank all candidates best-to-worst with hard code & NLI floors."""
        candidates = self.generator.generate_candidates(
            text, num_samples=self.num_samples, temperature=0.82, top_p=0.92, include_greedy=True
        )
        scored = self._score_candidates(text, candidates)
        seen_texts = {c.text for c in scored}

        strict_valid = [c for c in scored if self._passes_hard_constraints(c)]
        if not strict_valid:
            retry_n = max(3, self.num_samples // 2)
            retry_cands = self.generator.generate_candidates(
                text, num_samples=retry_n, temperature=0.3, top_p=0.9, include_greedy=True
            )
            new_cands = [c for c in retry_cands if c not in seen_texts]
            if new_cands:
                scored.extend(self._score_candidates(text, new_cands))
                seen_texts.update(new_cands)
            strict_valid = [c for c in scored if self._passes_hard_constraints(c)]

        if not strict_valid:
            fb_candidates: list[str] = []
            fb_text = self._get_classical_fallback().rewrite(text)
            if code_preservation_rate(text, fb_text) >= 1.0 and fb_text not in seen_texts:
                fb_candidates.append(fb_text)
            if text not in seen_texts:
                fb_candidates.append(text)
            if fb_candidates:
                scored.extend(self._score_candidates(text, fb_candidates))
                seen_texts.update(fb_candidates)
            strict_valid = [c for c in scored if self._passes_hard_constraints(c)]

        strict_valid.sort(key=lambda c: c.objective, reverse=True)
        code_only_valid = sorted(
            [
                c
                for c in scored
                if c.code_preservation >= 1.0 and not self._passes_hard_constraints(c)
            ],
            key=lambda c: c.objective,
            reverse=True,
        )
        invalid = sorted(
            [c for c in scored if c.code_preservation < 1.0],
            key=lambda c: c.objective,
            reverse=True,
        )
        return strict_valid + code_only_valid + invalid

    def rewrite_ranked(self, text: str) -> RankedCandidate:
        return self.rewrite_ranked_all(text)[0]

    def _rewrite_chunk(self, text: str) -> str:
        if self.skip_if_clear and is_already_clear(
            text, self.analyzer, threshold=self.clear_difficulty_threshold
        ):
            return text
        return self.rewrite_ranked(text).text

    def rewrite(self, text: str) -> str:
        return rewrite_markdown_preserving_structure(text, self._rewrite_chunk)


class LoRASinglePassReverser:
    """Single-pass instruct LLM + fine-tuned psycholinguistic reversal LoRA adapter."""

    def __init__(
        self,
        adapter_dir: str | Path,
        base_model: str | None = None,
        device: str | None = None,
        fallback_on_code_loss: bool = True,
        classical_fallback: ClassicalReverser | None = None,
        skip_if_clear: bool = True,
        clear_difficulty_threshold: float = 35.0,
        analyzer: Any | None = None,
    ):
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.adapter_dir = str(adapter_dir)
        if base_model is None:
            peft_cfg = PeftConfig.from_pretrained(self.adapter_dir)
            base_model = peft_cfg.base_model_name_or_path
            if not base_model:
                raise ValueError(
                    f"Could not auto-detect base_model_name_or_path from adapter '{self.adapter_dir}'. "
                    "Pass base_model explicitly."
                )
        self.base_model_name = base_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fallback_on_code_loss = fallback_on_code_loss
        self._classical_fallback = classical_fallback
        self.skip_if_clear = skip_if_clear
        self.clear_difficulty_threshold = float(clear_difficulty_threshold)
        self._analyzer = analyzer

        dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        if self.device != "cuda":
            dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model = PeftModel.from_pretrained(base, self.adapter_dir)
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.max_length = None
        self.model.eval()

    def _get_analyzer(self) -> Any:
        if self._analyzer is None:
            from psycho_rev.features import PsycholinguisticAnalyzer

            self._analyzer = PsycholinguisticAnalyzer()
        return self._analyzer

    def _get_classical_fallback(self) -> ClassicalReverser:
        if self._classical_fallback is None:
            self._classical_fallback = ClassicalReverser(safe_mode=True)
        return self._classical_fallback

    def _rewrite_chunk(self, text: str) -> str:
        import torch

        if self.skip_if_clear and is_already_clear(
            text, self._get_analyzer(), threshold=self.clear_difficulty_threshold
        ):
            return text

        guarded = protect_spans(text)
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Rewrite this passage clearly:\n\n{guarded.masked}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        max_new = min(max(int(len(guarded.masked.split()) * 2.4) + 32, 96), 384)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        result = _clean_output(guarded.restore(decoded))

        if self.fallback_on_code_loss and code_preservation_rate(text, result) < 1.0:
            with torch.inference_mode():
                retry_out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    num_return_sequences=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for seq in retry_out:
                cand_dec = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                cand = _clean_output(guarded.restore(cand_dec))
                if cand and code_preservation_rate(text, cand) >= 1.0:
                    return cand

            fb = self._get_classical_fallback().rewrite(text)
            if code_preservation_rate(text, fb) >= 1.0:
                return fb
            return text

        return result or text

    def rewrite(self, text: str) -> str:
        return rewrite_markdown_preserving_structure(text, self._rewrite_chunk)

    def close(self) -> None:
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HybridReverser:
    """Two-stage hybrid reverser: neural/API rewrite followed by ClassicalReverser post-processing."""

    def __init__(
        self,
        neural: Any,
        classical: ClassicalReverser | None = None,
        safe_mode: bool = True,
        fluency_gate: Callable[[str], float] | None = None,
        min_fluency_delta: float = -0.05,
    ):
        self.neural = neural
        self.classical = classical or ClassicalReverser(
            safe_mode=safe_mode,
            fluency_gate=fluency_gate,
            min_fluency_delta=min_fluency_delta,
        )

    def rewrite(self, text: str) -> str:
        return self.classical.rewrite(self.neural.rewrite(text))

    def close(self) -> None:
        if hasattr(self.neural, "close"):
            self.neural.close()
