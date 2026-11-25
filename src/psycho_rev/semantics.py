"""Semantic fidelity, fluency (GPT-2 PPL), and code preservation metrics."""

from __future__ import annotations

import gc
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from psycho_rev.code_guard import code_preservation_rate, protect_spans

_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def resolve_nli_label_indices(id2label: dict[Any, str] | None) -> tuple[int, int]:
    """Dynamically resolve (entailment_idx, contradiction_idx) from model id2label.

    Falls back to (1, 0) when id2label is missing or does not contain matching labels.
    """
    if not id2label:
        return (1, 0)

    entail_idx: int | None = None
    contra_idx: int | None = None
    for raw_idx, raw_label in id2label.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        label = str(raw_label).lower()
        if "entail" in label and entail_idx is None:
            entail_idx = idx
        elif "contradict" in label and contra_idx is None:
            contra_idx = idx

    return (
        entail_idx if entail_idx is not None else 1,
        contra_idx if contra_idx is not None else 0,
    )


def softmax_nli_probs(
    logits: np.ndarray | list[float],
    entail_idx: int = 1,
    contra_idx: int = 0,
) -> tuple[float, float]:
    """Compute (entailment_prob, contradiction_prob) via numerically stable softmax."""
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - np.max(arr)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps)
    return float(probs[entail_idx]), float(probs[contra_idx])


def _split_long_paragraph(paragraph: str, target_words: int = 150) -> list[str]:
    """Split a long paragraph into ~target_words chunks, preferring sentence boundaries."""
    words = paragraph.split()
    if len(words) <= target_words:
        return [paragraph.strip()]

    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(paragraph.strip()) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = sent.split()
        if len(sent_words) > target_words:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            for start in range(0, len(sent_words), target_words):
                chunks.append(" ".join(sent_words[start : start + target_words]))
            continue

        if current and current_words + len(sent_words) > target_words:
            chunks.append(" ".join(current))
            current = [sent]
            current_words = len(sent_words)
        else:
            current.append(sent)
            current_words += len(sent_words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_into_n_chunks(text: str, n_chunks: int) -> list[str]:
    """Partition text into n_chunks roughly equal word-count segments."""
    words = text.split()
    if n_chunks <= 1 or len(words) <= 1:
        return [text.strip()]
    n_chunks = min(n_chunks, len(words))
    base, rem = divmod(len(words), n_chunks)
    out: list[str] = []
    idx = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        out.append(" ".join(words[idx : idx + size]))
        idx += size
    return out


def align_text_chunks(
    original: str,
    rewritten: str,
    long_word_threshold: int = 200,
    chunk_word_target: int = 150,
) -> list[tuple[str, str]]:
    """Build aligned (orig_chunk, rew_chunk) pairs for NLI scoring.

    Short texts (<= long_word_threshold words) are scored as a single full-text pair.
    Longer texts are split into non-empty paragraphs (with long paragraphs further split
    into ~chunk_word_target-word chunks) and aligned so multi-paragraph passages are not
    silently truncated at 512 tokens.
    """
    if len(original.split()) <= long_word_threshold and len(rewritten.split()) <= long_word_threshold:
        return [(original, rewritten)]

    def _chunk_doc(doc: str) -> list[str]:
        paras = [p.strip() for p in _PARA_SPLIT_RE.split(doc.strip()) if p.strip()]
        if not paras:
            return [doc.strip()]
        chunks: list[str] = []
        for p in paras:
            chunks.extend(_split_long_paragraph(p, target_words=chunk_word_target))
        return chunks

    orig_chunks = _chunk_doc(original)
    rew_chunks = _chunk_doc(rewritten)

    if len(orig_chunks) == len(rew_chunks):
        return list(zip(orig_chunks, rew_chunks))

    target_n = max(len(orig_chunks), len(rew_chunks))
    if len(orig_chunks) != target_n:
        orig_chunks = _split_into_n_chunks(original, target_n)
    if len(rew_chunks) != target_n:
        rew_chunks = _split_into_n_chunks(rewritten, target_n)
    return list(zip(orig_chunks, rew_chunks))


def strip_fenced_code(text: str) -> str:
    """Remove markdown fenced code blocks so code tokens do not skew prose perplexity."""
    return _FENCED_CODE_RE.sub(" ", text).strip()


@dataclass
class SemanticReport:
    cosine_similarity: float
    nli_forward_entailment: float
    nli_backward_entailment: float
    nli_min_entailment: float
    gpt2_perplexity: float
    code_preservation: float
    length_ratio: float
    nli_max_contradiction: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticEvaluator:
    """Computes embedding cosine similarity, bidirectional NLI entailment, and GPT-2 fluency PPL."""

    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        fluency_model: str = "gpt2",
        device: str | None = None,
    ):
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(embed_model, device=self.device)
        self.nli = CrossEncoder(nli_model, device=self.device)
        id2label = getattr(
            getattr(getattr(self.nli, "model", None), "config", None),
            "id2label",
            None,
        )
        self._entail_idx, self._contra_idx = resolve_nli_label_indices(id2label)
        self.lm_tok = AutoTokenizer.from_pretrained(fluency_model)
        self.lm = AutoModelForCausalLM.from_pretrained(fluency_model).to(self.device)
        self.lm.eval()

    @staticmethod
    def _resolve_label_indices(id2label: dict[Any, str] | None) -> tuple[int, int]:
        return resolve_nli_label_indices(id2label)

    def _nli_probs(self, logits: np.ndarray) -> tuple[float, float]:
        return softmax_nli_probs(logits, self._entail_idx, self._contra_idx)

    def _entailment_prob(self, logits: np.ndarray) -> float:
        entail, _ = self._nli_probs(logits)
        return entail

    def _gpt2_ppl(self, text: str, max_length: int = 512, stride: int = 256) -> float:
        import torch

        prose = strip_fenced_code(text)
        if not prose:
            return 1.0
        clean = protect_spans(prose).masked
        enc = self.lm_tok(clean, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"]
        seq_len = int(input_ids.shape[1])
        if seq_len < 2:
            return 1.0

        if seq_len <= max_length:
            ids = input_ids.to(self.device)
            with torch.inference_mode():
                loss = self.lm(ids, labels=ids).loss.item()
            return round(min(math.exp(loss), 1e4), 2)

        nll_sum = 0.0
        total_tokens = 0
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            chunk_ids = input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = chunk_ids.clone()
            target_ids[:, :-trg_len] = -100

            num_loss_tokens = (trg_len - 1) if begin_loc == 0 else trg_len
            if num_loss_tokens > 0:
                with torch.inference_mode():
                    loss = self.lm(chunk_ids, labels=target_ids).loss.item()
                nll_sum += loss * num_loss_tokens
                total_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        avg_nll = nll_sum / max(total_tokens, 1)
        return round(min(math.exp(avg_nll), 1e4), 2)

    def evaluate(self, original: str, rewritten: str) -> SemanticReport:
        return self.evaluate_batch([original], [rewritten])[0]

    def evaluate_batch(self, originals: list[str], rewrittens: list[str]) -> list[SemanticReport]:
        if not originals:
            return []
        orig_embs = self.embedder.encode(originals, normalize_embeddings=True, batch_size=32)
        rew_embs = self.embedder.encode(rewrittens, normalize_embeddings=True, batch_size=32)
        cos_scores = np.sum(orig_embs * rew_embs, axis=1)

        pairs: list[tuple[str, str]] = []
        doc_spans: list[tuple[int, int]] = []
        for o, r in zip(originals, rewrittens):
            aligned = align_text_chunks(o, r)
            start = len(pairs)
            for oc, rc in aligned:
                pairs.append((oc, rc))
                pairs.append((rc, oc))
            doc_spans.append((start, len(pairs)))

        all_logits = self.nli.predict(pairs, batch_size=32)

        reports: list[SemanticReport] = []
        for i, (o, r) in enumerate(zip(originals, rewrittens)):
            start, end = doc_spans[i]
            fwd_entails: list[float] = []
            bwd_entails: list[float] = []
            contras: list[float] = []
            for idx in range(start, end, 2):
                f_ent, f_con = self._nli_probs(all_logits[idx])
                b_ent, b_con = self._nli_probs(all_logits[idx + 1])
                fwd_entails.append(f_ent)
                bwd_entails.append(b_ent)
                contras.append(f_con)
                contras.append(b_con)

            p_fwd = float(np.mean(fwd_entails)) if fwd_entails else 0.0
            p_bwd = float(np.mean(bwd_entails)) if bwd_entails else 0.0
            max_contra = float(max(contras)) if contras else 0.0
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
                    nli_max_contradiction=round(max_contra, 4),
                )
            )
        return reports

    def close(self) -> None:
        """Release loaded models and free GPU memory."""
        for attr in ("embedder", "nli", "lm", "lm_tok"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
