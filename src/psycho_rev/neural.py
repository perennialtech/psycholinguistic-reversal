"""Neural psycholinguistic reversal pipelines with open-weight models.

Implements four inference pipelines:
1. NLLBRoundTripReverser: English -> Japanese -> English bilingual pivoting (NLLB-200)
2. QwenZeroShotReverser: Single-pass open-weight instruct LLM rewriter
3. BestOfNPsycholinguisticReverser: Sample-and-rerank using psycholinguistic + semantic verifier
4. LoRASinglePassReverser: Single-pass Qwen2.5-1.5B-Instruct + fine-tuned LoRA adapter
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from psycho_rev.code_guard import code_preservation_rate, protect_spans
from psycho_rev.features import PsycholinguisticAnalyzer
from psycho_rev.semantics import SemanticEvaluator

REWRITE_SYSTEM_PROMPT = (
    "You are an editor who rewrites dense, hard-to-read technical and AI-style English into "
    "clear, natural, everyday human prose while preserving the exact meaning and all factual details.\n"
    "Rules:\n"
    "1. Keep every placeholder token like CODEITEM0, CODEITEM1 verbatim without altering or dropping it.\n"
    "2. Unpack stacked noun compounds (3+ nouns in a row) using clear prepositions.\n"
    "3. Replace ad-hoc denominal verbs (e.g., footnote, calendar, gate, surface, bucket, sunset, signpost, "
    "bottleneck, scaffold, sandbox, triage, workshop, anchor, bridge, spotlight, table) with plain verb phrases.\n"
    "4. Replace figurative animate verbs on inanimate subjects (e.g., eats, swallows, bites, chokes, starves, "
    "bleeds, refuses, punishes, betrays, sabotages) with literal, accurate verbs.\n"
    "5. Include explicit 'that' or 'which' in object relative clauses instead of omitting them.\n"
    "6. Shorten long syntactic dependencies by splitting overly complex clauses when helpful.\n"
    "7. Output ONLY the rewritten text, with no commentary, quotes, or markdown prefixes."
)


def _clean_output(text: str) -> str:
    out = text.strip()
    if len(out) >= 2 and out[0] == '"' and out[-1] == '"':
        out = out[1:-1].strip()
    return " ".join(out.split())


class NLLBRoundTripReverser:
    """Round-trip translation via Japanese using NLLB-200 (Clark et al. 2026 baseline)."""

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        pivot_lang: str = "jpn_Jpan",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pivot_lang = pivot_lang
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
            self.model.generation_config.max_length = None
        self.model.eval()

    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        tgt_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_id,
                max_new_tokens=256,
                num_beams=4,
            )
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

    def rewrite(self, text: str) -> str:
        guarded = protect_spans(text)
        ja = self._translate(guarded.masked, "eng_Latn", self.pivot_lang)
        en = self._translate(ja, self.pivot_lang, "eng_Latn")
        return _clean_output(guarded.restore(en))

    def close(self) -> None:
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
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
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
        if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
            self.model.generation_config.max_length = None
        self.model.eval()

    def _build_prompt(self, masked_text: str) -> str:
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Rewrite this passage clearly:\n\n{masked_text}"},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_candidates(
        self,
        text: str,
        num_samples: int = 1,
        temperature: float = 0.8,
        top_p: float = 0.92,
        include_greedy: bool = True,
    ) -> list[str]:
        guarded = protect_spans(text)
        prompt = self._build_prompt(guarded.masked)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        max_new = min(max(int(len(guarded.masked.split()) * 2.4) + 32, 96), 320)

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

        # Deduplicate preserving order
        deduped: list[str] = []
        seen: set[str] = set()
        for r in results:
            if r and r not in seen:
                seen.add(r)
                deduped.append(r)
        return deduped or [text]

    def rewrite(self, text: str) -> str:
        return self.generate_candidates(text, num_samples=1, include_greedy=True)[0]

    def close(self) -> None:
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


class BestOfNPsycholinguisticReverser:
    """Inference-time search: sample K rewrites and select the best psycholinguistic + semantic trade-off."""

    def __init__(
        self,
        generator: QwenZeroShotReverser,
        analyzer: PsycholinguisticAnalyzer,
        semantics: SemanticEvaluator,
        num_samples: int = 6,
    ):
        self.generator = generator
        self.analyzer = analyzer
        self.semantics = semantics
        self.num_samples = num_samples

    def rewrite_ranked(self, text: str) -> RankedCandidate:
        candidates = self.generator.generate_candidates(
            text, num_samples=self.num_samples, temperature=0.82, top_p=0.92, include_greedy=True
        )
        sem_reports = self.semantics.evaluate_batch([text] * len(candidates), candidates)

        scored: list[RankedCandidate] = []
        for cand, sem in zip(candidates, sem_reports):
            feat = self.analyzer.analyze(cand)
            len_pen = max(0.0, sem.length_ratio - 1.65) * 12.0 + max(0.0, 0.75 - sem.length_ratio) * 20.0
            code_pen = (1.0 - sem.code_preservation) * 80.0
            # Penalize hallucination / semantic drift heavily
            sem_bonus = 45.0 * sem.cosine_similarity + 40.0 * sem.nli_min_entailment
            obj = -feat.difficulty_score + sem_bonus - len_pen - code_pen
            scored.append(
                RankedCandidate(
                    text=cand,
                    objective=round(obj, 3),
                    difficulty_score=feat.difficulty_score,
                    cosine_similarity=sem.cosine_similarity,
                    nli_min_entailment=sem.nli_min_entailment,
                    code_preservation=sem.code_preservation,
                )
            )
        scored.sort(key=lambda c: c.objective, reverse=True)
        return scored[0]

    def rewrite(self, text: str) -> str:
        return self.rewrite_ranked(text).text


class LoRASinglePassReverser:
    """Single-pass Qwen2.5-1.5B-Instruct + fine-tuned psycholinguistic reversal LoRA."""

    def __init__(
        self,
        adapter_dir: str | Path,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
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
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
            self.model.generation_config.max_length = None
        self.model.eval()

    def rewrite(self, text: str) -> str:
        guarded = protect_spans(text)
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Rewrite this passage clearly:\n\n{guarded.masked}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        max_new = min(max(int(len(guarded.masked.split()) * 2.4) + 32, 96), 320)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        return _clean_output(guarded.restore(decoded))

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
