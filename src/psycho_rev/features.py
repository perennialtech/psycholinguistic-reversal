"""Psycholinguistic difficulty feature detectors based on Clark, Martin, & Mahowald (2026).

Measures the six features analyzed in arXiv:2610.29162v1:
1. N-gram statistics (unattested bigram rate, trigram perplexity vs. human corpus)
2. Standardized syntactic dependency length
3. Object relative clauses (reduced vs. overt relativizer)
4. Stacked noun compounds (3+ nouns in a modifier chain)
5. Denominal verbs (ad-hoc noun-to-verb conversions)
6. Inanimate subjects paired with animate-preferring active verbs
"""

from __future__ import annotations

import math
import os
import pickle
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import spacy
from nltk.corpus import brown, reuters, wordnet as wn

from psycho_rev.code_guard import protect_spans

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "psycholinguistic-reversal"

# Verbs that strongly prefer animate agents (selectional restriction violations with inanimate nsubj)
ANIMATE_VERBS: set[str] = {
    "argue", "betray", "bite", "bleed", "breathe", "chew", "choke", "choose",
    "cling", "complain", "crawl", "cry", "dance", "decide", "demand", "desire",
    "digest", "eat", "embrace", "fear", "feel", "fight", "fool", "forgive",
    "glance", "grab", "groan", "hate", "hear", "hug", "hunger", "insist",
    "kiss", "know", "laugh", "lie", "listen", "love", "negotiate", "obey",
    "pretend", "promise", "punish", "rebel", "refuse", "reward", "sabotage",
    "scream", "seize", "shout", "sigh", "sleep", "smell", "speak", "sprint",
    "stare", "starve", "struggle", "stumble", "swallow", "taste", "think",
    "trick", "trip", "wait", "wake", "walk", "want", "watch", "weep",
    "whisper", "wrestle",
}

# Known ad-hoc / AI-favored denominal verbs (including examples from Clark et al. 2026 & LLM corpora)
KNOWN_DENOMINAL_VERBS: set[str] = {
    "anchor", "architect", "benchmark", "bottleneck", "bridge", "bucket",
    "calendar", "champion", "containerize", "eyeball", "fast-track", "footnote",
    "funnel", "gate", "guardrail", "hand-wave", "handwave", "headquarter",
    "impact", "instrument", "leverage", "mirror", "orchestrate",
    "paperwork", "pipeline", "platform", "productize", "rubber-stamp",
    "sandbox", "scaffold", "shadow", "shelf", "shelve", "shoulder", "signpost",
    "silo", "spearhead", "spotlight", "stonewall", "sunset", "surface",
    "table", "task", "template", "triage", "weaponize", "whiteboard",
    "workshop",
}

# Frequent conventional verbs excluded from WordNet-based denominal heuristics
CORE_NON_DENOMINAL_VERBS: set[str] = {
    "add", "allow", "appear", "apply", "ask", "avoid", "be", "become", "begin",
    "believe", "bring", "build", "buy", "call", "can", "carry", "catch", "cause",
    "change", "check", "clear", "close", "come", "compare", "complete", "consider",
    "contain", "continue", "control", "copy", "cost", "could", "count", "cover",
    "create", "cut", "deal", "define", "delete", "deliver", "depend", "describe",
    "design", "detect", "determine", "develop", "die", "display", "do", "draw",
    "drive", "drop", "enable", "end", "ensure", "enter", "exist", "expect",
    "explain", "express", "extend", "face", "fail", "fall", "feed", "fill",
    "find", "finish", "fix", "flow", "focus", "follow", "force", "form", "gain",
    "generate", "get", "give", "go", "grow", "handle", "happen", "have", "head",
    "help", "hide", "hit", "hold", "hope", "identify", "ignore", "improve",
    "include", "increase", "indicate", "insert", "install", "introduce", "involve",
    "issue", "join", "keep", "kill", "lead", "learn", "leave", "let", "lift",
    "like", "limit", "link", "list", "live", "load", "lock", "log", "look",
    "lose", "maintain", "make", "manage", "map", "mark", "match", "matter",
    "may", "mean", "measure", "meet", "merge", "might", "miss", "modify", "move",
    "must", "name", "need", "note", "notice", "occur", "offer", "open", "operate",
    "order", "output", "own", "pack", "parse", "part", "pass", "pay", "perform",
    "pick", "place", "plan", "play", "point", "post", "prefer", "prepare",
    "present", "preserve", "prevent", "print", "process", "produce", "program",
    "protect", "prove", "provide", "pull", "push", "put", "query", "raise",
    "reach", "read", "realize", "receive", "recognize", "recommend", "record",
    "recover", "reduce", "refer", "reflect", "register", "reject", "relate",
    "release", "rely", "remain", "remember", "remove", "render", "replace",
    "reply", "report", "represent", "request", "require", "reset", "resolve",
    "respond", "rest", "restore", "restrict", "result", "retain", "return",
    "reveal", "review", "rewrite", "rise", "roll", "route", "run", "save", "say",
    "scale", "scan", "schedule", "score", "search", "see", "seek", "seem",
    "select", "send", "separate", "serve", "set", "settle", "shall", "shape",
    "share", "shift", "shoot", "should", "show", "shut", "sign", "simplify",
    "sit", "skip", "solve", "sort", "sound", "spend", "split", "spread", "stand",
    "start", "state", "stay", "step", "stick", "stop", "store", "stream",
    "strike", "strip", "study", "submit", "succeed", "suffer", "suggest", "suit",
    "summarize", "supply", "support", "suppose", "switch", "sync", "take", "talk",
    "target", "teach", "tell", "tend", "test", "thank", "throw", "tie", "time",
    "touch", "trace", "track", "trade", "train", "transfer", "transform",
    "translate", "treat", "trigger", "trust", "try", "turn", "type", "understand",
    "update", "upgrade", "upload", "use", "validate", "value", "vary", "verify",
    "view", "visit", "vote", "warn", "wash", "waste", "wear", "weigh", "will",
    "win", "wish", "wonder", "work", "worry", "would", "wrap", "write", "yield",
}

ANIMATE_PRONOUNS: set[str] = {
    "i", "you", "he", "she", "we", "they", "who", "whoever", "someone",
    "somebody", "anyone", "anybody", "everyone", "everybody", "nobody", "noone",
}

ANIMATE_HYPERNYMS: set[str] = {
    "person.n.01", "animal.n.01", "organism.n.01", "living_thing.n.01",
    "people.n.01", "social_group.n.01", "organization.n.01",
}

OVERT_RELATIVIZERS: set[str] = {"that", "which", "who", "whom", "whose", "where", "when", "why"}

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


class ReferenceNgramModel:
    """Human-corpus bigram/trigram reference model built from NLTK Brown + Reuters."""

    def __init__(self, unigrams: Counter[str], bigrams: Counter[tuple[str, str]], trigrams: Counter[tuple[str, str, str]], total_tokens: int):
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.trigrams = trigrams
        self.total_tokens = total_tokens
        self.vocab_size = len(unigrams)

    @classmethod
    def load_or_build(cls) -> "ReferenceNgramModel":
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / "ngram_ref_v1.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return cls(**data)

        unigrams: Counter[str] = Counter()
        bigrams: Counter[tuple[str, str]] = Counter()
        trigrams: Counter[tuple[str, str, str]] = Counter()
        total = 0

        for corpus in (brown, reuters):
            for sent in corpus.sents():
                toks = ["<s>", "<s>"] + [w.lower() for w in sent if _WORD_RE.fullmatch(w.lower())] + ["</s>"]
                if len(toks) <= 3:
                    continue
                for i in range(2, len(toks)):
                    w0, w1, w2 = toks[i - 2], toks[i - 1], toks[i]
                    unigrams[w2] += 1
                    bigrams[(w1, w2)] += 1
                    trigrams[(w0, w1, w2)] += 1
                    total += 1

        payload = {
            "unigrams": unigrams,
            "bigrams": bigrams,
            "trigrams": trigrams,
            "total_tokens": total,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f)
        return cls(**payload)

    def score_text(self, text: str) -> tuple[float, float]:
        """Return (unattested_bigram_rate, trigram_perplexity) for text."""
        clean = protect_spans(text).masked
        sents = re.split(r"[.!?;\n]+", clean)
        bigram_total = 0
        unattested = 0
        log_prob_sum = 0.0
        n_scored = 0

        v = max(self.vocab_size, 1)
        total = max(self.total_tokens, 1)

        for sent in sents:
            words = [w for w in _WORD_RE.findall(sent.lower()) if not w.startswith("codeitem")]
            if len(words) < 2:
                continue
            for i in range(len(words) - 1):
                bigram_total += 1
                if (words[i], words[i + 1]) not in self.bigrams:
                    unattested += 1

            padded = ["<s>", "<s>"] + words + ["</s>"]
            for i in range(2, len(padded)):
                w0, w1, w2 = padded[i - 2], padded[i - 1], padded[i]
                c_tri = self.trigrams.get((w0, w1, w2), 0)
                c_bi_ctx = self.bigrams.get((w0, w1), 0)
                c_bi = self.bigrams.get((w1, w2), 0)
                c_uni_ctx = self.unigrams.get(w1, 0)
                c_uni = self.unigrams.get(w2, 0)

                p3 = (c_tri / c_bi_ctx) if c_bi_ctx > 0 else 0.0
                p2 = (c_bi / c_uni_ctx) if c_uni_ctx > 0 else 0.0
                p1 = (c_uni + 0.5) / (total + 0.5 * v)
                p = 0.65 * p3 + 0.25 * p2 + 0.10 * p1
                log_prob_sum += math.log(max(p, 1e-12))
                n_scored += 1

        unattested_rate = (unattested / bigram_total) if bigram_total > 0 else 0.0
        ppl = math.exp(-log_prob_sum / n_scored) if n_scored > 0 else 1.0
        return unattested_rate, min(ppl, 1e6)


@dataclass
class FeatureReport:
    word_count: int
    sentence_count: int
    unattested_bigram_rate: float
    trigram_perplexity: float
    mean_dep_length: float
    std_dep_length: float
    reduced_orc_count: int
    full_orc_count: int
    reduced_orc_per_100w: float
    full_orc_per_100w: float
    stacked_noun_count: int
    stacked_noun_per_100w: float
    denominal_verb_count: int
    denominal_verb_per_100w: float
    inanimate_subj_count: int
    inanimate_subj_per_100w: float
    difficulty_score: float
    matches: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_inanimate_noun(token: spacy.tokens.Token) -> bool:
    low = token.text.lower()
    lemma = token.lemma_.lower()
    if low in ANIMATE_PRONOUNS or lemma in ANIMATE_PRONOUNS:
        return False
    if token.ent_type_ in {"PERSON", "ORG", "NORP"}:
        return False
    if token.pos_ == "PRON":
        return low in {"it", "this", "that", "these", "those", "which", "something", "nothing", "everything"}
    if token.pos_ not in {"NOUN", "PROPN"}:
        return False
    if low.startswith("codeitem"):
        return True
    synsets = wn.synsets(lemma, pos=wn.NOUN)
    if not synsets:
        return True
    for syn in synsets[:3]:
        for path in syn.hypernym_paths():
            if any(h.name() in ANIMATE_HYPERNYMS for h in path):
                return False
    return True


def _is_denominal_verb(token: spacy.tokens.Token) -> bool:
    if token.pos_ != "VERB":
        return False
    lemma = token.lemma_.lower()
    if lemma in CORE_NON_DENOMINAL_VERBS or len(lemma) < 3:
        return False
    if lemma in KNOWN_DENOMINAL_VERBS:
        return True

    noun_syns = wn.synsets(lemma, pos=wn.NOUN)
    verb_syns = wn.synsets(lemma, pos=wn.VERB)
    if not noun_syns:
        return False
    noun_freq = sum(l.count() for s in noun_syns for l in s.lemmas() if l.name().lower() == lemma)
    verb_freq = sum(l.count() for s in verb_syns for l in s.lemmas() if l.name().lower() == lemma)
    has_obj = any(c.dep_ in {"dobj", "obj", "nsubjpass"} for c in token.children)
    if has_obj and noun_freq >= 10 and (verb_freq == 0 or noun_freq / max(verb_freq, 1) >= 8.0):
        return True
    return False


def _detect_orcs(doc: spacy.tokens.Doc) -> tuple[list[str], list[str]]:
    """Return (reduced_orc_spans, full_orc_spans)."""
    reduced: list[str] = []
    full: list[str] = []
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
        has_relativizer = any(
            t.text.lower() in OVERT_RELATIVIZERS and t.dep_ in {"dobj", "obj", "pobj", "mark", "advmod", "det"}
            for t in between
        ) or any(
            t.text.lower() in OVERT_RELATIVIZERS and t.i == head.i + 1
            for t in doc[head.i : min(len(doc), tok.i + 1)]
        )
        snippet = doc[head.i : min(len(doc), tok.i + 2)].text
        if has_relativizer:
            full.append(snippet)
        else:
            reduced.append(snippet)
    return reduced, full


def _detect_stacked_nouns(doc: spacy.tokens.Doc) -> list[str]:
    """Detect maximal stacked noun compounds of length >= 3 (including hyphenated noun modifiers)."""
    stacks: list[str] = []
    visited: set[int] = set()
    for tok in doc:
        if tok.i in visited:
            continue
        if tok.pos_ not in {"NOUN", "PROPN"}:
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
            for c in chain:
                visited.add(c.i)
            span_text = doc[chain[0].i : chain[-1].i + 1].text
            if "CODEITEM" not in span_text:
                stacks.append(span_text)
    return stacks


class PsycholinguisticAnalyzer:
    """Computes all six psycholinguistic difficulty metrics and composite score."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        self.ngram = ReferenceNgramModel.load_or_build()

    def analyze(self, text: str) -> FeatureReport:
        guarded = protect_spans(text)
        doc = self.nlp(guarded.masked)

        words = [t for t in doc if not t.is_punct and not t.is_space]
        word_count = max(len(words), 1)
        sents = list(doc.sents)

        unattested_rate, trigram_ppl = self.ngram.score_text(text)

        dep_lengths: list[int] = []
        std_lengths: list[float] = []
        for sent in sents:
            sent_toks = [t for t in sent if not t.is_punct and not t.is_space]
            if len(sent_toks) < 3:
                continue
            s_dists = [abs(t.i - t.head.i) for t in sent_toks if t.head != t]
            if s_dists:
                mean_d = sum(s_dists) / len(s_dists)
                dep_lengths.extend(s_dists)
                expected_baseline = max((len(sent_toks) + 1) / 3.0, 1.0)
                std_lengths.append(mean_d / expected_baseline)

        mean_dep = sum(dep_lengths) / len(dep_lengths) if dep_lengths else 0.0
        std_dep = sum(std_lengths) / len(std_lengths) if std_lengths else 0.0

        reduced_orcs, full_orcs = _detect_orcs(doc)
        stacked_nouns = _detect_stacked_nouns(doc)
        denominals = [t.text for t in doc if _is_denominal_verb(t)]

        inanimate_matches: list[str] = []
        for tok in doc:
            if tok.pos_ != "VERB" or tok.lemma_.lower() not in ANIMATE_VERBS:
                continue
            for child in tok.children:
                if child.dep_ == "nsubj" and _is_inanimate_noun(child):
                    inanimate_matches.append(f"{guarded.restore(child.text)} {tok.text}")

        scale = 100.0 / word_count
        red_orc_rate = len(reduced_orcs) * scale
        full_orc_rate = len(full_orcs) * scale
        stacked_rate = len(stacked_nouns) * scale
        denom_rate = len(denominals) * scale
        inanim_rate = len(inanimate_matches) * scale

        ppl_term = min(math.log10(max(trigram_ppl, 1.0)) / 4.0, 1.5) * 15.0
        difficulty_score = (
            unattested_rate * 25.0
            + ppl_term
            + max(mean_dep - 2.0, 0.0) * 8.0
            + red_orc_rate * 4.0
            + stacked_rate * 5.0
            + denom_rate * 6.0
            + inanim_rate * 6.0
        )

        return FeatureReport(
            word_count=len(words),
            sentence_count=len(sents),
            unattested_bigram_rate=round(unattested_rate, 4),
            trigram_perplexity=round(trigram_ppl, 2),
            mean_dep_length=round(mean_dep, 3),
            std_dep_length=round(std_dep, 3),
            reduced_orc_count=len(reduced_orcs),
            full_orc_count=len(full_orcs),
            reduced_orc_per_100w=round(red_orc_rate, 2),
            full_orc_per_100w=round(full_orc_rate, 2),
            stacked_noun_count=len(stacked_nouns),
            stacked_noun_per_100w=round(stacked_rate, 2),
            denominal_verb_count=len(denominals),
            denominal_verb_per_100w=round(denom_rate, 2),
            inanimate_subj_count=len(inanimate_matches),
            inanimate_subj_per_100w=round(inanim_rate, 2),
            difficulty_score=round(difficulty_score, 2),
            matches={
                "reduced_orcs": reduced_orcs,
                "full_orcs": full_orcs,
                "stacked_nouns": stacked_nouns,
                "denominal_verbs": denominals,
                "inanimate_subjects": inanimate_matches,
            },
        )
