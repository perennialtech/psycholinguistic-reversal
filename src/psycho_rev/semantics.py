"""Semantic fidelity, fluency (GPT-2 PPL), and code preservation metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from psycho_rev.code_guard import code_preservation_rate, protect_spans


@dataclass
class SemanticReport:
    cosine_similarity: float
    nli_forward_entailment: float
    nli_backward_entailment: float
    nli_min_entailment: float
    gpt2_perplexity: float
    code_preservation: float
    length_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticEvaluator:
    """Computes embedding cosine similarity, bidirectional NLI entailment, and GPT-2 fluency PPL."""

    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model: str = "cross-encoder/nli-deberta-v3-small",
        fluency_model: str = "gpt2",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(embed_model, device=self.device)
        self.nli = CrossEncoder(nli_model, device=self.device)
        self.lm_tok = AutoTokenizer.from_pretrained(fluency_model)
        self.lm = AutoModelForCausalLM.from_pretrained(fluency_model).to(self.device)
        self.lm.eval()

    @staticmethod
    def _entailment_prob(logits: np.ndarray) -> float:
        shifted = logits - np.max(logits)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        return float(probs[1])

    def _gpt2_ppl(self, text: str) -> float:
        clean = protect_spans(text).masked
        enc = self.lm_tok(clean, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        if enc["input_ids"].shape[1] < 2:
            return 1.0
        with torch.inference_mode():
            loss = self.lm(**enc, labels=enc["input_ids"]).loss.item()
        return round(min(math.exp(loss), 1e4), 2)

    def evaluate(self, original: str, rewritten: str) -> SemanticReport:
        return self.evaluate_batch([original], [rewritten])[0]

    def evaluate_batch(self, originals: list[str], rewrittens: list[str]) -> list[SemanticReport]:
        if not originals:
            return []
        orig_embs = self.embedder.encode(originals, normalize_embeddings=True, batch_size=32)
        rew_embs = self.embedder.encode(rewrittens, normalize_embeddings=True, batch_size=32)
        cos_scores = np.sum(orig_embs * rew_embs, axis=1)

        pairs: list[tuple[str, str]] = []
        for o, r in zip(originals, rewrittens):
            pairs.append((o, r))
            pairs.append((r, o))
        all_logits = self.nli.predict(pairs, batch_size=32)

        reports: list[SemanticReport] = []
        for i, (o, r) in enumerate(zip(originals, rewrittens)):
            p_fwd = self._entailment_prob(all_logits[2 * i])
            p_bwd = self._entailment_prob(all_logits[2 * i + 1])
            orig_words = max(len(o.split()), 1)
            reports.append(
                SemanticReport(
                    cosine_similarity=round(float(cos_scores[i]), 4),
                    nli_forward_entailment=round(p_fwd, 4),
                    nli_backward_entailment=round(p_bwd, 4),
                    nli_min_entailment=round(min(p_fwd, p_bwd), 4),
                    gpt2_perplexity=self._gpt2_ppl(r),
                    code_preservation=round(code_preservation_rate(o, r), 4),
                    length_ratio=round(len(r.split()) / orig_words, 3),
                )
            )
        return reports
