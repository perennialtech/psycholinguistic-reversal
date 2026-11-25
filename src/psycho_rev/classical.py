"""Classical (symbolic / spaCy dependency-tree + lexical frame) psycholinguistic reverser.

Targets the six features from Clark, Martin, & Mahowald (2026) without neural generation:
1. Reduced object relative clauses -> insert overt 'that' relativizer
2. Denominal verbs -> rewrite via curated light-verb + preposition frames
3. Inanimate subjects with animate verbs -> replace figurative animate verb with literal verb
4. Stacked noun compounds (3+ nouns) -> heuristic rightmost-head prepositional unpacking
5. Long dependency clauses -> split long coordinated clauses on explicit conjunctions
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import spacy

from psycho_rev.code_guard import protect_spans
from psycho_rev.features import (
    ANIMATE_VERBS,
    OVERT_RELATIVIZERS,
    _is_inanimate_noun,
)

# (base, 3sg, past, gerund) literal replacements for figurative animate verbs with inanimate subjects
INANIMATE_VERB_FRAMES: dict[str, tuple[str, str, str, str]] = {
    "eat": ("consume", "consumes", "consumed", "consuming"),
    "swallow": ("silently ignore", "silently ignores", "silently ignored", "silently ignoring"),
    "bite": ("cause problems", "causes problems", "caused problems", "causing problems"),
    "choke": ("fail", "fails", "failed", "failing"),
    "starve": ("deprive", "deprives", "deprived", "depriving"),
    "bleed": ("leak", "leaks", "leaked", "leaking"),
    "refuse": ("fail", "fails", "failed", "failing"),
    "demand": ("require", "requires", "required", "requiring"),
    "insist": ("require", "requires", "required", "requiring"),
    "punish": ("penalize", "penalizes", "penalized", "penalizing"),
    "betray": ("expose", "exposes", "exposed", "exposing"),
    "sabotage": ("disrupt", "disrupts", "disrupted", "disrupting"),
    "whisper": ("indicate subtly", "indicates subtly", "indicated subtly", "indicating subtly"),
    "scream": ("strongly indicate", "strongly indicates", "strongly indicated", "strongly indicating"),
    "stumble": ("fail", "fails", "failed", "failing"),
    "wrestle": ("struggle", "struggles", "struggled", "struggling"),
    "hunger": ("require", "requires", "required", "requiring"),
    "cheat": ("bypass", "bypasses", "bypassed", "bypassing"),
    "trick": ("mislead", "misleads", "misled", "misleading"),
    "fool": ("mislead", "misleads", "misled", "misleading"),
}

# (base, 3sg, past, gerund) expansions for common AI-favored denominal verbs
DENOMINAL_VERB_FRAMES: dict[str, tuple[str, str, str, str]] = {
    "footnote": ("add a footnote to", "adds a footnote to", "added a footnote to", "adding a footnote to"),
    "calendar": ("put on the calendar", "puts on the calendar", "put on the calendar", "putting on the calendar"),
    "gate": ("restrict", "restricts", "restricted", "restricting"),
    "surface": ("expose", "exposes", "exposed", "exposing"),
    "bucket": ("group", "groups", "grouped", "grouping"),
    "scaffold": ("set up the structure for", "sets up the structure for", "set up the structure for", "setting up the structure for"),
    "sunset": ("phase out", "phases out", "phased out", "phasing out"),
    "signpost": ("clearly mark", "clearly marks", "clearly marked", "clearly marking"),
    "bottleneck": ("slow down", "slows down", "slowed down", "slowing down"),
    "anchor": ("base", "bases", "based", "basing"),
    "workshop": ("refine", "refines", "refined", "refining"),
    "sandbox": ("isolate", "isolates", "isolated", "isolating"),
    "triage": ("prioritize", "prioritizes", "prioritized", "prioritizing"),
    "mirror": ("match", "matches", "matched", "matching"),
    "bridge": ("connect", "connects", "connected", "connecting"),
    "funnel": ("route", "routes", "routed", "routing"),
    "champion": ("advocate for", "advocates for", "advocated for", "advocating for"),
    "architect": ("design", "designs", "designed", "designing"),
    "shoulder": ("take on", "takes on", "took on", "taking on"),
    "silo": ("isolate", "isolates", "isolated", "isolating"),
    "spearhead": ("lead", "leads", "led", "leading"),
    "spotlight": ("highlight", "highlights", "highlighted", "highlighting"),
    "benchmark": ("measure", "measures", "measured", "measuring"),
    "instrument": ("add telemetry to", "adds telemetry to", "added telemetry to", "adding telemetry to"),
    "leverage": ("use", "uses", "used", "using"),
    "pipeline": ("chain", "chains", "chained", "chaining"),
    "shadow": ("duplicate", "duplicates", "duplicated", "duplicating"),
    "shelve": ("postpone", "postpones", "postponed", "postponing"),
    "table": ("postpone", "postpones", "postponed", "postponing"),
    "eyeball": ("estimate visually", "estimates visually", "estimated visually", "estimating visually"),
}


def _inflect_frame(token: spacy.tokens.Token, forms: tuple[str, str, str, str]) -> str:
    base, third_sg, past, gerund = forms
    tag = token.tag_
    if tag == "VBZ":
        choice = third_sg
    elif tag in {"VBD", "VBN"}:
        choice = past
    elif tag == "VBG":
        choice = gerund
    else:
        choice = base
    if token.text and token.text[0].isupper():
        return choice[0].upper() + choice[1:]
    return choice


@dataclass
class ClassicalRewriteResult:
    text: str
    edits_applied: dict[str, int]


class ClassicalReverser:
    """Deterministic spaCy + lexicon reverser for psycholinguistic difficulty features."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)

    def rewrite_with_stats(self, text: str) -> ClassicalRewriteResult:
        guarded = protect_spans(text)
        doc = self.nlp(guarded.masked)

        edits = {
            "orc_that_inserted": 0,
            "denominal_expanded": 0,
            "inanimate_verb_replaced": 0,
            "noun_stack_unpacked": 0,
            "clause_split": 0,
        }

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

        # 2. Expand known denominal verbs & 3. Replace animate verbs on inanimate subjects
        for tok in doc:
            if tok.pos_ != "VERB":
                continue
            lemma = tok.lemma_.lower()
            if lemma in DENOMINAL_VERB_FRAMES:
                replacements[tok.i] = _inflect_frame(tok, DENOMINAL_VERB_FRAMES[lemma])
                edits["denominal_expanded"] += 1
                continue
            if lemma in ANIMATE_VERBS and lemma in INANIMATE_VERB_FRAMES:
                has_inanim_subj = any(
                    c.dep_ == "nsubj" and _is_inanimate_noun(c) for c in tok.children
                )
                if has_inanim_subj:
                    replacements[tok.i] = _inflect_frame(tok, INANIMATE_VERB_FRAMES[lemma])
                    edits["inanimate_verb_replaced"] += 1

        # 4. Unpack 3+ stacked noun compounds: [N1 N2 N3] -> [N3 for N1 N2]
        visited_stacks: set[int] = set()
        for tok in doc:
            if tok.i in visited_stacks or tok.pos_ not in {"NOUN", "PROPN"}:
                continue
            chain = [tok]
            cur = tok.i
            while cur + 1 < len(doc):
                nxt = doc[cur + 1]
                if nxt.pos_ in {"NOUN", "PROPN"} and (
                    nxt.dep_ == "compound"
                    or chain[-1].dep_ == "compound"
                    or (nxt.head == chain[-1] or chain[-1].head == nxt)
                ):
                    chain.append(nxt)
                    cur += 1
                elif nxt.text == "-" and cur + 2 < len(doc) and doc[cur + 2].pos_ in {"NOUN", "PROPN"}:
                    chain.append(doc[cur + 2])
                    cur += 2
                else:
                    break
            if len(chain) >= 3:
                span_start = chain[0].i
                span_end = chain[-1].i
                span_text = doc[span_start : span_end + 1].text
                if "CODEITEM" in span_text or any(i in replacements for i in range(span_start, span_end + 1)):
                    continue
                head_noun = chain[-1].text
                modifiers = " ".join(t.text for t in chain[:-1])
                unpacked = f"{head_noun} for {modifiers}"
                if doc[span_start].text and doc[span_start].text[0].isupper() and head_noun[0].islower():
                    unpacked = unpacked[0].upper() + unpacked[1:]
                replacements[span_start] = unpacked
                for idx in range(span_start + 1, span_end + 1):
                    skip_indices.add(idx)
                for c in chain:
                    visited_stacks.add(c.i)
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
                    # Split only if preceded by comma and followed by a clause with its own subject
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
            # If we replaced a multi-token span, take trailing whitespace from last skipped token
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
        return ClassicalRewriteResult(
            text=guarded.restore(reconstructed),
            edits_applied=edits,
        )

    def rewrite(self, text: str) -> str:
        return self.rewrite_with_stats(text).text
