"""Semantic fidelity, fluency (LM PPL), grammatical acceptability (CoLA), and code preservation metrics."""

from __future__ import annotations

import gc
import logging
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from psycho_rev.code_guard import code_preservation_rate, protect_spans

logger = logging.getLogger(__name__)

_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~", re.MULTILINE)
_YAML_FRONTMATTER_RE = re.compile(r"\A[ \t]*\n*---[ \t]*\n[\s\S]*?\n---[ \t]*(?:\n|\Z)")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MD_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+(.*)$")
_MD_LIST_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+(.*)$")
_MD_BLOCKQUOTE_RE = re.compile(r"^[ \t]*>[ \t]?(.*)$")
_MD_TABLE_LINE_RE = re.compile(r"^[ \t]*\|.*$")
_MD_TABLE_SEP_RE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$"
)
_MD_THEMATIC_BREAK_RE = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$")
_MD_PREFIX_RE = re.compile(r"^[ \t]*(?:#{1,6}[ \t]+|(?:[-*+]|\d+[.)])[ \t]+|>[ \t]*)")
_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

DEFAULT_EVAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EVAL_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
DEFAULT_EVAL_FLUENCY_MODEL = "gpt2"
DEFAULT_HELDOUT_FLUENCY_MODEL = "gpt2-medium"
DEFAULT_RERANK_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_RERANK_FLUENCY_MODEL = "gpt2"
DEFAULT_COLA_MODEL = "textattack/roberta-base-CoLA"


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


def resolve_cola_label_index(id2label: dict[Any, str] | None) -> int:
    """Resolve the index of the grammatically acceptable class from CoLA model id2label.

    Standard CoLA checkpoints (including ``textattack/roberta-base-CoLA``) map
    ``0 -> unacceptable`` and ``1 -> acceptable``. Falls back to ``1``.
    """
    if not id2label:
        return 1

    acceptable_idx: int | None = None
    unacceptable_idx: int | None = None
    indices: list[int] = []
    for raw_idx, raw_label in id2label.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        indices.append(idx)
        label = str(raw_label).strip().lower()
        if "unacceptable" in label or "incorrect" in label or "invalid" in label:
            unacceptable_idx = idx
        elif "acceptable" in label or "correct" in label or "valid" in label:
            acceptable_idx = idx

    if acceptable_idx is not None:
        return acceptable_idx
    if unacceptable_idx is not None and len(indices) == 2:
        other = [i for i in indices if i != unacceptable_idx]
        if other:
            return other[0]
    return 1


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


def softmax_positive_prob(logits: np.ndarray | list[float], positive_idx: int = 1) -> float:
    """Compute probability of ``positive_idx`` via numerically stable softmax."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 0:
        return float(1.0 / (1.0 + math.exp(-float(arr))))
    if arr.size == 1:
        return float(1.0 / (1.0 + math.exp(-float(arr.flat[0]))))
    shifted = arr - np.max(arr)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps)
    idx = min(max(int(positive_idx), 0), len(probs) - 1)
    return float(np.clip(probs[idx], 0.0, 1.0))


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
    """Remove markdown fenced code blocks so code tokens do not skew prose metrics."""
    return _FENCED_CODE_RE.sub(" ", text).strip()


def split_prose_sentences(text: str) -> list[str]:
    """Extract clean prose sentences for sentence-level grammatical acceptability scoring.

    Strips YAML frontmatter, fenced code blocks, Markdown pipe tables, and leading
    heading/list/blockquote markers; joins soft-wrapped prose lines within the same
    paragraph; masks inline code spans and technical identifiers; and splits on
    sentence boundaries.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    normalized = _YAML_FRONTMATTER_RE.sub("", normalized, count=1)
    prose = _FENCED_CODE_RE.sub("\n\n", normalized).strip()
    if not prose:
        return []

    blocks: list[str] = []
    para_lines: list[str] = []
    in_blockquote = False

    def _flush_para() -> None:
        if para_lines:
            blocks.append(" ".join(para_lines))
            para_lines.clear()

    for raw_line in prose.splitlines():
        if not raw_line.strip():
            _flush_para()
            in_blockquote = False
            continue

        if _MD_TABLE_LINE_RE.match(raw_line) or _MD_TABLE_SEP_RE.match(raw_line):
            _flush_para()
            in_blockquote = False
            continue

        if _MD_THEMATIC_BREAK_RE.match(raw_line):
            _flush_para()
            in_blockquote = False
            continue

        heading_match = _MD_HEADING_RE.match(raw_line)
        if heading_match:
            _flush_para()
            in_blockquote = False
            heading_text = re.sub(r"[ \t]+#+[ \t]*$", "", heading_match.group(1)).strip()
            if heading_text:
                blocks.append(heading_text)
            continue

        bq_match = _MD_BLOCKQUOTE_RE.match(raw_line)
        if bq_match:
            if not in_blockquote:
                _flush_para()
                in_blockquote = True
            inner = bq_match.group(1).strip()
            if not inner:
                _flush_para()
                continue
            if _MD_TABLE_LINE_RE.match(inner) or _MD_TABLE_SEP_RE.match(inner):
                _flush_para()
                continue
            list_match = _MD_LIST_RE.match(inner)
            if list_match:
                _flush_para()
                item_text = list_match.group(1).strip()
                if item_text:
                    para_lines.append(item_text)
            else:
                para_lines.append(inner)
            continue

        if in_blockquote:
            _flush_para()
            in_blockquote = False

        list_match = _MD_LIST_RE.match(raw_line)
        if list_match:
            _flush_para()
            item_text = list_match.group(1).strip()
            if item_text:
                para_lines.append(item_text)
            continue

        para_lines.append(raw_line.strip())

    _flush_para()

    sentences: list[str] = []
    for block in blocks:
        masked = protect_spans(block).masked
        for part in _SENT_SPLIT_RE.split(masked):
            sent = part.strip()
            if not sent:
                continue
            tokens = _ALPHA_TOKEN_RE.findall(sent)
            if len(tokens) >= 2:
                sentences.append(sent)
    return sentences


class AcceptabilityScorer:
    """Standalone lazy-loaded CoLA grammatical acceptability scorer.

    Evaluates text sentence-by-sentence and returns both the mean and minimum
    acceptability probability in ``[0.0, 1.0]`` so a single ungrammatical sentence
    inside an otherwise fluent passage is surfaced rather than averaged away.

    Can be imported independently by classical or neural rewriters as a lightweight
    grammaticality gate via ``is_acceptable(...)`` or ``score_text(...)``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_COLA_MODEL,
        device: str | None = None,
        enabled: bool = True,
        batch_size: int = 32,
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.device = device
        self.enabled = enabled
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(16, int(max_length))
        self._tokenizer: Any = None
        self._model: Any = None
        self._acceptable_idx: int = 1
        self._load_failed: bool = False
        self._load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _ensure_loaded(self) -> bool:
        if not self.enabled:
            return False
        if self._model is not None and self._tokenizer is not None:
            return True
        if self._load_failed:
            return False

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = resolved_device
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(
                resolved_device
            )
            self._model.eval()
            id2label = getattr(getattr(self._model, "config", None), "id2label", None)
            self._acceptable_idx = resolve_cola_label_index(id2label)
            return True
        except Exception as exc:
            self._load_failed = True
            self._load_error = str(exc)
            logger.warning(
                "CoLA acceptability model %r unavailable (%s); falling back to neutral 1.0.",
                self.model_name,
                exc,
            )
            return False

    def score_sentences(self, sentences: list[str]) -> list[float]:
        """Return acceptability probabilities in [0.0, 1.0] for each sentence."""
        if not sentences:
            return []
        if not self._ensure_loaded():
            return [1.0] * len(sentences)

        import torch

        scores: list[float] = []
        for start in range(0, len(sentences), self.batch_size):
            batch = sentences[start : start + self.batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if self.device is not None and hasattr(enc, "to"):
                enc = enc.to(self.device)
            elif self.device is not None and isinstance(enc, dict):
                enc = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in enc.items()}

            with torch.inference_mode():
                outputs = self._model(**enc)
                logits = outputs.logits

            if hasattr(logits, "detach"):
                logits = logits.detach()
            if hasattr(logits, "cpu"):
                logits = logits.cpu()
            logits_arr = np.asarray(logits, dtype=np.float64)
            if logits_arr.ndim == 1:
                logits_arr = logits_arr[None, :]
            for row in logits_arr:
                scores.append(softmax_positive_prob(row, positive_idx=self._acceptable_idx))
        return scores

    def score_text(self, text: str) -> tuple[float, float]:
        """Return ``(acceptability_mean, acceptability_min)`` for a single passage."""
        return self.score_texts([text])[0]

    def score_texts(self, texts: list[str]) -> list[tuple[float, float]]:
        """Batch-score passages, returning ``(acceptability_mean, acceptability_min)`` per text."""
        if not texts:
            return []
        if not self.enabled:
            return [(1.0, 1.0)] * len(texts)

        all_sentences: list[str] = []
        spans: list[tuple[int, int]] = []
        for text in texts:
            sents = split_prose_sentences(text)
            start = len(all_sentences)
            all_sentences.extend(sents)
            spans.append((start, len(all_sentences)))

        if not all_sentences:
            return [(1.0, 1.0)] * len(texts)

        flat_scores = self.score_sentences(all_sentences)
        results: list[tuple[float, float]] = []
        for start, end in spans:
            if start == end:
                results.append((1.0, 1.0))
                continue
            seg = flat_scores[start:end]
            results.append((round(float(np.mean(seg)), 4), round(float(np.min(seg)), 4)))
        return results

    def is_acceptable(self, text: str, min_threshold: float = 0.5) -> bool:
        """Return True when every sentence meets ``min_threshold`` grammatical acceptability."""
        _, acc_min = self.score_text(text)
        return acc_min >= min_threshold

    def close(self) -> None:
        """Release loaded CoLA model weights and free GPU memory."""
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


@dataclass
class SemanticReport:
    """Semantic fidelity, fluency, code preservation, and grammatical acceptability metrics.

    ``acceptability_score`` mirrors ``acceptability_min`` (worst-sentence CoLA grammatical
    acceptability probability in ``[0.0, 1.0]``) so a single mangled sentence is never
    hidden by averaging over surrounding fluent sentences.
    """

    cosine_similarity: float
    nli_forward_entailment: float
    nli_backward_entailment: float
    nli_min_entailment: float
    gpt2_perplexity: float
    code_preservation: float
    length_ratio: float
    nli_max_contradiction: float = 0.0
    acceptability_score: float = 1.0
    acceptability_mean: float = 1.0
    acceptability_min: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticEvaluator:
    """Computes embedding cosine similarity, bidirectional NLI, LM fluency PPL, and CoLA acceptability.

    Supports ``role="eval"`` (default held-out configuration using ``nli-deberta-v3-base``)
    and ``role="rerank"`` (fast proxy configuration defaulting to ``nli-deberta-v3-small``)
    so candidate reranking can be decoupled from held-out evaluation.
    """

    def __init__(
        self,
        embed_model: str = DEFAULT_EVAL_EMBED_MODEL,
        nli_model: str = DEFAULT_EVAL_NLI_MODEL,
        fluency_model: str = DEFAULT_EVAL_FLUENCY_MODEL,
        device: str | None = None,
        *,
        role: str = "eval",
        enable_acceptability: bool = True,
        acceptability_model: str = DEFAULT_COLA_MODEL,
        lm_model: str | None = None,
    ):
        if role not in {"eval", "rerank"}:
            raise ValueError(f"Unsupported role {role!r}; expected 'eval' or 'rerank'.")

        resolved_nli_model = nli_model
        if role == "rerank" and nli_model == DEFAULT_EVAL_NLI_MODEL:
            resolved_nli_model = DEFAULT_RERANK_NLI_MODEL
        resolved_lm_model = lm_model if lm_model is not None else fluency_model

        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.role = role
        self.embed_model_name = embed_model
        self.nli_model_name = resolved_nli_model
        self.fluency_model_name = resolved_lm_model
        self.enable_acceptability = enable_acceptability

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(embed_model, device=self.device)
        self.nli = CrossEncoder(resolved_nli_model, device=self.device)
        id2label = getattr(
            getattr(getattr(self.nli, "model", None), "config", None),
            "id2label",
            None,
        )
        self._entail_idx, self._contra_idx = resolve_nli_label_indices(id2label)
        self.lm_tok = AutoTokenizer.from_pretrained(resolved_lm_model)
        self.lm = AutoModelForCausalLM.from_pretrained(resolved_lm_model).to(self.device)
        self.lm.eval()
        self.acceptability_scorer: AcceptabilityScorer | None = AcceptabilityScorer(
            model_name=acceptability_model,
            device=self.device,
            enabled=enable_acceptability,
        )

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

        scorer = getattr(self, "acceptability_scorer", None)
        if scorer is not None and getattr(self, "enable_acceptability", True):
            acc_results = scorer.score_texts(rewrittens)
        else:
            acc_results = [(1.0, 1.0)] * len(rewrittens)

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
            acc_mean, acc_min = acc_results[i]

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
                    acceptability_score=acc_min,
                    acceptability_mean=acc_mean,
                    acceptability_min=acc_min,
                )
            )
        return reports

    def close(self) -> None:
        """Release loaded models and free GPU memory."""
        scorer = getattr(self, "acceptability_scorer", None)
        if scorer is not None:
            scorer.close()
        for attr in ("embedder", "nli", "lm", "lm_tok", "acceptability_scorer"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class HeldOutSemanticEvaluator(SemanticEvaluator):
    """Independent held-out semantic evaluator decoupled from reranker proxy models.

    Defaults to ``cross-encoder/nli-deberta-v3-base`` for bidirectional NLI,
    ``textattack/roberta-base-CoLA`` for sentence-level grammatical acceptability,
    and ``gpt2-medium`` (configurable via ``lm_model``) for fluency perplexity.
    """

    def __init__(
        self,
        embed_model: str = DEFAULT_EVAL_EMBED_MODEL,
        nli_model: str = DEFAULT_EVAL_NLI_MODEL,
        lm_model: str = DEFAULT_HELDOUT_FLUENCY_MODEL,
        acceptability_model: str = DEFAULT_COLA_MODEL,
        enable_acceptability: bool = True,
        device: str | None = None,
    ):
        super().__init__(
            embed_model=embed_model,
            nli_model=nli_model,
            fluency_model=lm_model,
            device=device,
            role="eval",
            enable_acceptability=enable_acceptability,
            acceptability_model=acceptability_model,
            lm_model=lm_model,
        )
