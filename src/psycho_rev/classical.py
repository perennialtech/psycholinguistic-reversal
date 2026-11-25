"""Classical (symbolic / spaCy dependency-tree + lexical frame) psycholinguistic reverser.

Targets the six features from Clark, Martin, & Mahowald (2026) without neural generation:
1. Reduced object relative clauses -> insert overt 'that' relativizer
2. Denominal verbs -> rewrite via curated light-verb + preposition frames
3. Inanimate subjects with animate verbs -> replace figurative animate verb with literal verb
4. Stacked noun compounds (3+ nouns) -> head-class prepositional unpacking
5. Long dependency clauses -> split long coordinated clauses on explicit conjunctions
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import spacy

from psycho_rev.code_guard import protect_spans, rewrite_markdown_preserving_structure
from psycho_rev.features import (
    ANIMATE_VERBS,
    OVERT_RELATIVIZERS,
    _is_inanimate_noun,
    _normalized_denominal_lemma,
    find_stacked_noun_spans,
)

# (base, 3sg, past, past_participle, gerund) literal replacements for figurative animate verbs
INANIMATE_VERB_FRAMES: dict[str, tuple[str, str, str, str, str]] = {
    "eat": ("consume", "consumes", "consumed", "consumed", "consuming"),
    "swallow": ("silently ignore", "silently ignores", "silently ignored", "silently ignored", "silently ignoring"),
    "bite": ("cause problems", "causes problems", "caused problems", "caused problems", "causing problems"),
    "choke": ("fail", "fails", "failed", "failed", "failing"),
    "starve": ("deprive", "deprives", "deprived", "deprived", "depriving"),
    "bleed": ("leak", "leaks", "leaked", "leaked", "leaking"),
    "refuse": ("fail", "fails", "failed", "failed", "failing"),
    "demand": ("require", "requires", "required", "required", "requiring"),
    "insist": ("require", "requires", "required", "required", "requiring"),
    "punish": ("penalize", "penalizes", "penalized", "penalized", "penalizing"),
    "betray": ("expose", "exposes", "exposed", "exposed", "exposing"),
    "sabotage": ("disrupt", "disrupts", "disrupted", "disrupted", "disrupting"),
    "whisper": ("indicate subtly", "indicates subtly", "indicated subtly", "indicated subtly", "indicating subtly"),
    "scream": ("strongly indicate", "strongly indicates", "strongly indicated", "strongly indicated", "strongly indicating"),
    "stumble": ("fail", "fails", "failed", "failed", "failing"),
    "wrestle": ("struggle", "struggles", "struggled", "struggled", "struggling"),
    "hunger": ("require", "requires", "required", "required", "requiring"),
    "cheat": ("bypass", "bypasses", "bypassed", "bypassed", "bypassing"),
    "trick": ("mislead", "misleads", "misled", "misled", "misleading"),
    "fool": ("mislead", "misleads", "misled", "misled", "misleading"),
}

# (base, 3sg, past, past_participle, gerund) expansions for common AI-favored denominal verbs
DENOMINAL_VERB_FRAMES: dict[str, tuple[str, str, str, str, str]] = {
    "footnote": ("add a footnote to", "adds a footnote to", "added a footnote to", "footnoted", "adding a footnote to"),
    "calendar": ("put on the calendar", "puts on the calendar", "put on the calendar", "put on the calendar", "putting on the calendar"),
    "gate": ("restrict", "restricts", "restricted", "restricted", "restricting"),
    "surface": ("expose", "exposes", "exposed", "exposed", "exposing"),
    "bucket": ("group", "groups", "grouped", "grouped", "grouping"),
    "scaffold": ("set up the structure for", "sets up the structure for", "set up the structure for", "set up", "setting up the structure for"),
    "sunset": ("phase out", "phases out", "phased out", "phased out", "phasing out"),
    "signpost": ("clearly mark", "clearly marks", "clearly marked", "clearly marked", "clearly marking"),
    "bottleneck": ("slow down", "slows down", "slowed down", "slowed down", "slowing down"),
    "anchor": ("base", "bases", "based", "based", "basing"),
    "workshop": ("refine", "refines", "refined", "refined", "refining"),
    "sandbox": ("isolate", "isolates", "isolated", "isolated", "isolating"),
    "triage": ("prioritize", "prioritizes", "prioritized", "prioritized", "prioritizing"),
    "mirror": ("match", "matches", "matched", "matched", "matching"),
    "bridge": ("connect", "connects", "connected", "connected", "connecting"),
    "funnel": ("route", "routes", "routed", "routed", "routing"),
    "champion": ("advocate for", "advocates for", "advocated for", "championed", "advocating for"),
    "architect": ("design", "designs", "designed", "designed", "designing"),
    "shoulder": ("take on", "takes on", "took on", "taken on", "taking on"),
    "silo": ("isolate", "isolates", "isolated", "isolated", "isolating"),
    "spearhead": ("lead", "leads", "led", "led", "leading"),
    "spotlight": ("highlight", "highlights", "highlighted", "highlighted", "highlighting"),
    "benchmark": ("measure", "measures", "measured", "measured", "measuring"),
    "instrument": ("add telemetry to", "adds telemetry to", "added telemetry to", "instrumented", "adding telemetry to"),
    "leverage": ("use", "uses", "used", "used", "using"),
    "pipeline": ("chain", "chains", "chained", "chained", "chaining"),
    "shadow": ("duplicate", "duplicates", "duplicated", "duplicated", "duplicating"),
    "shelve": ("postpone", "postpones", "postponed", "postponed", "postponing"),
    "table": ("postpone", "postpones", "postponed", "postponed", "postponing"),
    "eyeball": ("estimate visually", "estimates visually", "estimated visually", "estimated visually", "estimating visually"),
}

# Denominal verbs that have common literal intransitive or animate-object senses
# and must only be rewritten in transitive/passive/abstract contexts.
_TRANSITIVE_GATED_DENOMINALS: set[str] = {
    "anchor", "bridge", "bucket", "funnel", "gate", "mirror",
    "sandbox", "shadow", "shoulder", "surface", "table",
}

# Head noun classes for natural stacked-noun prepositional unpacking
_IN_HEAD_NOUNS: set[str] = {
    "adoption", "area", "bottleneck", "change", "corridor", "decline", "deficit",
    "delay", "drift", "drop", "gap", "growth", "increase", "investment", "market",
    "participation", "pause", "reduction", "region", "rise", "role", "sector", "shift",
}

_OF_HEAD_NOUNS: set[str] = {
    "allocation", "assumption", "audit", "author", "backlog", "bandwidth", "bias",
    "boundary", "cadence", "capacity", "ceiling", "collection", "committee",
    "compaction", "consumer", "count", "cycle", "digitization", "distribution",
    "ecosystem", "edge", "eviction", "exposure", "floor", "format", "goodwill",
    "guarantee", "hierarchy", "infrastructure", "invalidation", "invariant",
    "kind", "latency", "layer", "length", "level", "lift", "limit", "load",
    "log", "manager", "member", "metric", "migration", "modernization", "momentum",
    "network", "offset", "overhead", "overview", "owner", "part", "penalty",
    "phase", "queue", "rate", "ratio", "restructuring", "review", "rollback",
    "rollout", "score", "size", "snapshot", "stack", "stage", "state", "statement",
    "status", "step", "structure", "summary", "threshold", "throughput", "trace",
    "type", "update", "velocity", "version", "volume", "window",
}

_VOWEL_START_RE = re.compile(r"^[aeiouAEIOU]")


def _inflect_frame(token: spacy.tokens.Token, forms: tuple[str, str, str, str, str]) -> str:
    base, third_sg, past, past_part, gerund = forms
    tag = token.tag_
    is_passive = any(c.dep_ == "auxpass" for c in token.children) or (
        token.dep_ in {"acl", "amod"} and tag == "VBN"
    )
    if tag == "VBZ":
        choice = third_sg
    elif tag == "VBN" or is_passive:
        choice = past_part
    elif tag == "VBD":
        choice = past
    elif tag == "VBG":
        choice = gerund
    else:
        choice = base
    if token.text and token.text[0].isupper():
        return choice[0].upper() + choice[1:]
    return choice


def _should_rewrite_denominal(token: spacy.tokens.Token, lemma: str) -> bool:
    """Context-gate denominal verb rewrites to avoid mangling literal conventional uses."""
    if lemma not in DENOMINAL_VERB_FRAMES:
        return False

    # Avoid rewriting passive/participle forms where our frame would be a no-op
    is_passive = any(c.dep_ == "auxpass" for c in token.children) or token.tag_ == "VBN"
    if is_passive and DENOMINAL_VERB_FRAMES[lemma][3] == token.text.lower():
        return False

    if lemma not in _TRANSITIVE_GATED_DENOMINALS:
        return True

    objs = [c for c in token.children if c.dep_ in {"dobj", "obj", "ccomp", "xcomp"}]
    has_passive_subj = any(c.dep_ in {"nsubjpass", "auxpass"} for c in token.children)
    is_rel_or_acl = token.dep_ in {"relcl", "acl"}

    # Special case: 'shadow' with an animate direct object (e.g. "shadowed the engineer") is literal
    if lemma == "shadow" and objs and not _is_inanimate_noun(objs[0]):
        return False

    # Special case: 'anchor ... on <abstract>' is denominal even if spaCy attaches object differently
    if lemma == "anchor" and any(c.dep_ == "prep" and c.text.lower() == "on" for c in token.children):
        return True

    return bool(objs or has_passive_subj or is_rel_or_acl)


def _choose_stack_preposition(head_tok: spacy.tokens.Token) -> str:
    """Select a natural preposition ('of', 'in', 'for') based on the head noun class."""
    lemma = head_tok.lemma_.lower()
    if lemma in _IN_HEAD_NOUNS:
        return "in"
    if lemma in _OF_HEAD_NOUNS:
        return "of"
    return "for"


def _unpack_stacked_noun(doc: spacy.tokens.Doc, start_i: int, end_i: int) -> str:
    """Unpack a stacked noun span doc[start_i:end_i+1] using a head-appropriate preposition."""
    head_tok = doc[end_i]
    head_noun = head_tok.text
    prep = _choose_stack_preposition(head_tok)

    # Preserve internal hyphenation in the modifier prefix
    mod_parts: list[str] = []
    for idx in range(start_i, end_i):
        t = doc[idx]
        text = t.text
        if idx == start_i and text and text[0].isupper() and t.pos_ != "PROPN":
            text = text[0].lower() + text[1:]
        ws = t.whitespace_ if idx < end_i - 1 else ""
        mod_parts.append(text + ws)
    modifiers = "".join(mod_parts).strip()

    unpacked = f"{head_noun} {prep} {modifiers}"
    if doc[start_i].text and doc[start_i].text[0].isupper() and head_noun and head_noun[0].islower():
        unpacked = unpacked[0].upper() + unpacked[1:]
    return unpacked


@dataclass
class ClassicalRewriteResult:
    text: str
    edits_applied: dict[str, int]


class ClassicalReverser:
    """Deterministic spaCy + lexicon reverser for psycholinguistic difficulty features."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)

    def _rewrite_segment(self, text: str, edits: dict[str, int]) -> str:
        guarded = protect_spans(text)
        doc = self.nlp(guarded.masked)

        # Map token index -> replacement string (or prefix insertion)
        replacements: dict[int, str] = {}
        insert_before: dict[int, str] = {}
        skip_indices: set[int] = set()

        # 1. Insert 'that' into reduced object relative clauses
        for tok in doc:
            if tok.dep_ != "relcl":
                continue
            head = tok.head
            if head.pos_ not in {"NOUN", "PROPN", "PRON"}:
                continue
            subjs = [c for c in tok.children if c.dep_ in {"nsubj", "nsubjpass"}]
            if not subjs:
                continue
            subj = subjs[0]
            if subj.tag_ in {"WDT", "WP", "WP$"} or subj.text.lower() in OVERT_RELATIVIZERS:
                continue
            span_start = min(head.i + 1, tok.i)
            span_end = max(head.i + 1, tok.i + 1)
            between = doc[span_start:span_end]
            has_rel = any(t.text.lower() in OVERT_RELATIVIZERS for t in between)
            if not has_rel and head.i + 1 < len(doc):
                insert_before[head.i + 1] = "that "
                edits["orc_that_inserted"] += 1

        # 2. Expand context-gated denominal verbs & 3. Replace animate verbs on inanimate subjects
        for tok in doc:
            if tok.pos_ != "VERB":
                continue
            lemma = _normalized_denominal_lemma(tok)
            if _should_rewrite_denominal(tok, lemma):
                replacements[tok.i] = _inflect_frame(tok, DENOMINAL_VERB_FRAMES[lemma])
                edits["denominal_expanded"] += 1
                continue
            raw_lemma = tok.lemma_.lower()
            if raw_lemma in ANIMATE_VERBS and raw_lemma in INANIMATE_VERB_FRAMES:
                has_inanim_subj = any(
                    c.dep_ == "nsubj" and _is_inanimate_noun(c) for c in tok.children
                )
                if has_inanim_subj:
                    replacements[tok.i] = _inflect_frame(tok, INANIMATE_VERB_FRAMES[raw_lemma])
                    edits["inanimate_verb_replaced"] += 1

        # 4. Unpack 3+ stacked noun compounds using shared detector from features.py
        for span_start, span_end, span_text in find_stacked_noun_spans(doc):
            if "CODEITEM" in span_text or any(i in replacements for i in range(span_start, span_end + 1)):
                continue
            unpacked = _unpack_stacked_noun(doc, span_start, span_end)
            replacements[span_start] = unpacked
            for idx in range(span_start + 1, span_end + 1):
                skip_indices.add(idx)

            # Fix indefinite article agreement ('a' vs 'an') immediately preceding the unpacked span
            if span_start > 0 and (span_start - 1) not in replacements and (span_start - 1) not in skip_indices:
                prev_tok = doc[span_start - 1]
                prev_low = prev_tok.text.lower()
                starts_vowel = bool(_VOWEL_START_RE.match(unpacked))
                if prev_low == "an" and not starts_vowel:
                    replacements[prev_tok.i] = "A" if prev_tok.text[0].isupper() else "a"
                elif prev_low == "a" and starts_vowel:
                    replacements[prev_tok.i] = "An" if prev_tok.text[0].isupper() else "an"

            edits["noun_stack_unpacked"] += 1

        # 5. Split overly long sentences (>26 words) at high-level clause conjunctions
        for sent in doc.sents:
            sent_words = [t for t in sent if not t.is_punct and not t.is_space]
            if len(sent_words) <= 26:
                continue
            for tok in sent:
                if (
                    tok.dep_ == "cc"
                    and tok.text.lower() in {"and", "but"}
                    and tok.i > sent.start + 8
                    and tok.i < sent.end - 6
                ):
                    prev_tok = doc[tok.i - 1]
                    conj_verb = tok.head
                    has_conj_subj = any(
                        c.dep_ == "conj" and any(gc.dep_ in {"nsubj", "nsubjpass"} for gc in c.children)
                        for c in conj_verb.children
                    )
                    if prev_tok.text == "," and has_conj_subj and prev_tok.i not in skip_indices:
                        replacements[prev_tok.i] = "."
                        skip_indices.add(tok.i)
                        nxt = doc[tok.i + 1]
                        if nxt.i not in replacements and nxt.i not in skip_indices and nxt.text:
                            replacements[nxt.i] = nxt.text[0].upper() + nxt.text[1:]
                        edits["clause_split"] += 1
                        break

        # Reconstruct text preserving original whitespace
        out_parts: list[str] = []
        for tok in doc:
            if tok.i in skip_indices:
                continue
            prefix = insert_before.get(tok.i, "")
            body = replacements.get(tok.i, tok.text)
            ws = tok.whitespace_
            if tok.i in replacements and tok.i + 1 in skip_indices:
                last = tok.i
                while last + 1 in skip_indices:
                    last += 1
                ws = doc[last].whitespace_
            out_parts.append(prefix + body + ws)

        reconstructed = "".join(out_parts)
        reconstructed = re.sub(r"\s+([.,;:!?])", r"\1", reconstructed)
        reconstructed = re.sub(r" {2,}", " ", reconstructed).strip()
        return guarded.restore(reconstructed)

    def rewrite_with_stats(self, text: str) -> ClassicalRewriteResult:
        edits = {
            "orc_that_inserted": 0,
            "denominal_expanded": 0,
            "inanimate_verb_replaced": 0,
            "noun_stack_unpacked": 0,
            "clause_split": 0,
        }
        rewritten = rewrite_markdown_preserving_structure(
            text,
            lambda seg: self._rewrite_segment(seg, edits),
        )
        return ClassicalRewriteResult(
            text=rewritten,
            edits_applied=edits,
        )

    def rewrite(self, text: str) -> str:
        return self.rewrite_with_stats(text).text
