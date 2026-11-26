"""Resumable real-world LLM corpus generator, OOD splitter, and teacher-distillation pipeline.

Generates authentic uncurated prose from real LLMs across four realistic domains
(coding, technical_docs, essay, workplace_chat), constructs deduplicated and
out-of-lexicon tagged splits, and distills teacher rewrites + hard negatives
(including identity/no-op examples) for SFT/DPO/SimPO training.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import random
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx

from psycho_rev.classical import DENOMINAL_VERB_FRAMES, INANIMATE_VERB_FRAMES
from psycho_rev.code_guard import (
    code_preservation_rate,
    protect_spans,
    strip_fenced_code_blocks,
)
from psycho_rev.datasets import (
    DEFAULT_DATASET_DIR,
    DatasetSample,
    DistilledPair,
    load_jsonl,
    save_jsonl,
)
from psycho_rev.features import DENOMINAL_SURFACE_TO_LEMMA

logger = logging.getLogger(__name__)

DOMAINS: tuple[str, ...] = ("coding", "technical_docs", "essay", "workplace_chat")

DEFAULT_GENERATION_MODELS: tuple[str, ...] = (
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "meta-llama/llama-3.3-70b-instruct",
)
DEFAULT_TEACHER_MODEL = "anthropic/claude-3.5-sonnet"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"

# Approximate (input_usd_per_1m, output_usd_per_1m) pricing for budget enforcement.
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "gpt-4o": (2.50, 10.00),
    "anthropic/claude-3.5-haiku": (0.80, 4.00),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3.7-sonnet": (3.00, 15.00),
    "meta-llama/llama-3.3-70b-instruct": (0.20, 0.60),
    "google/gemini-2.0-flash-001": (0.10, 0.40),
    "qwen/qwen-2.5-72b-instruct": (0.30, 0.90),
}
DEFAULT_PRICING_PER_1M: tuple[float, float] = (1.00, 3.00)

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_FENCE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")


@dataclass(frozen=True)
class PromptSpec:
    """A single natural real-world prompt for generating authentic LLM prose."""

    prompt_id: str
    domain: str
    style: str  # "single_para" or "multi_para_markdown"
    prompt: str
    expects_code: bool = False


def _build_prompt_bank() -> tuple[PromptSpec, ...]:
    """Construct 120 distinct, natural real-world prompts (30 per domain)."""
    specs: list[PromptSpec] = []

    coding_prompts: list[tuple[str, str, bool]] = [
        ("single_para", "Explain why mutating a Python list while iterating over it skips elements, referencing a short inline snippet like `for item in items:`.", True),
        ("single_para", "Describe what causes a stale closure bug inside a React `useEffect` hook and how adding the missing dependency fixes it.", True),
        ("single_para", "Explain why comparing floating-point totals with `==` fails in billing code and recommend a cleaner `Decimal` alternative.", True),
        ("single_para", "Summarize why using `SELECT *` in production PostgreSQL queries breaks rolling schema migrations.", True),
        ("single_para", "Explain the practical difference between `asyncio.gather()` and `asyncio.TaskGroup` when one coroutine raises an exception.", True),
        ("single_para", "Describe why storing JWTs in `localStorage` creates XSS exposure compared to `HttpOnly` cookies.", True),
        ("single_para", "Explain what happens when a Go goroutine writes to a nil map or sends on a closed channel.", True),
        ("single_para", "Explain why SQLite raises `database is locked` under concurrent writes and how enabling WAL mode helps.", True),
        ("single_para", "Describe why Rust rejects returning a reference to a local variable from a function and how returning an owned `String` solves it.", True),
        ("single_para", "Explain why `subprocess.run(..., shell=True)` is risky with untrusted input and show the list-argument pattern inline.", True),
        ("single_para", "Describe what an N+1 query problem looks like in an ORM endpoint and how eager loading eliminates it.", True),
        ("single_para", "Explain why Python's `datetime.now()` without a timezone leads to subtle bugs across daylight saving transitions.", True),
        ("single_para", "Explain why `git rebase` on a shared branch disrupts teammates and when `git merge` is safer.", True),
        ("single_para", "Describe why HTTP `429` retry loops need exponential backoff with jitter rather than a fixed one-second sleep.", True),
        ("single_para", "Explain why unindexed foreign keys slow down `DELETE` statements on parent tables in PostgreSQL.", True),
        ("multi_para_markdown", "Write a short two-paragraph code review explanation with a small Python fenced block showing why catching bare `Exception` hides `KeyboardInterrupt` or cancellation errors and how to narrow the handler.", True),
        ("multi_para_markdown", "Write a two-paragraph explanation with a fenced SQL example showing how a composite index `(tenant_id, created_at DESC)` speeds up paginated inbox queries.", True),
        ("multi_para_markdown", "Write a concise debugging note with a fenced Bash snippet explaining why missing `set -euo pipefail` in `scripts/deploy.sh` lets failed build steps go unnoticed.", True),
        ("multi_para_markdown", "Explain in two short paragraphs with a fenced TypeScript example how discriminated unions eliminate unsafe type assertions in API response handlers.", True),
        ("multi_para_markdown", "Write a two-paragraph note with a fenced Python example showing how `contextlib.ExitStack` cleanly manages a dynamic number of open file handles.", True),
        ("multi_para_markdown", "Explain with a short Dockerfile snippet why placing `COPY . .` before `pip install -r requirements.txt` invalidates Docker layer caching on every edit.", True),
        ("multi_para_markdown", "Write a two-paragraph explanation with a fenced Go snippet showing why forgetting `defer resp.Body.Close()` leaks TCP connections under load.", True),
        ("multi_para_markdown", "Explain in two paragraphs with a fenced Python snippet why default mutable arguments like `def append_item(x, buf=[]):` share state across calls.", True),
        ("multi_para_markdown", "Write a short markdown note with a fenced SQL query explaining how `SELECT ... FOR UPDATE SKIP LOCKED` prevents double-processing in a job queue.", True),
        ("multi_para_markdown", "Explain with a short Rust fenced snippet how `Option::ok_or_else` avoids eagerly allocating an error string on the happy path.", True),
        ("multi_para_markdown", "Write a two-paragraph explanation with a fenced YAML snippet showing how Kubernetes `readinessProbe` differs from `livenessProbe` during slow startup.", True),
        ("multi_para_markdown", "Explain in two paragraphs with a fenced Python snippet why streaming large CSV exports with a generator keeps worker memory flat.", True),
        ("multi_para_markdown", "Write a short code review note with a fenced JavaScript snippet explaining why `Promise.allSettled` is better than `Promise.all` when sending independent webhook notifications.", True),
        ("multi_para_markdown", "Explain in two paragraphs with a fenced SQL snippet why `OFFSET` pagination degrades on deep pages and how keyset pagination stays fast.", True),
        ("multi_para_markdown", "Write a two-paragraph debugging guide with a fenced Python snippet showing how `hmac.compare_digest` prevents timing side channels in webhook verification.", True),
    ]

    technical_docs_prompts: list[tuple[str, str, bool]] = [
        ("single_para", "Write a one-paragraph overview for an API documentation page explaining how idempotency keys prevent duplicate payment charges on client retries.", True),
        ("single_para", "Draft a concise changelog entry explaining that `v2.4.0` deprecates the `timeout_ms` query parameter in favor of the `X-Request-Timeout` header.", True),
        ("single_para", "Write a short configuration reference paragraph explaining how `MAX_POOL_SIZE` and `IDLE_TIMEOUT_SEC` interact in a backend database service.", True),
        ("single_para", "Draft a one-paragraph security note explaining how TLS certificate rotation works with zero downtime behind an ingress proxy.", False),
        ("single_para", "Write a brief architecture overview paragraph describing how a write-ahead log guarantees crash recovery in a key-value store.", False),
        ("single_para", "Write a concise troubleshooting paragraph explaining why Prometheus scrape targets show `context deadline exceeded` during garbage collection pauses.", True),
        ("single_para", "Draft a one-paragraph SDK migration note explaining how to move from callback-based file uploads to async iterators.", True),
        ("single_para", "Write a short operational note describing when to trigger a manual read-replica failover versus waiting for automated promotion.", False),
        ("single_para", "Draft a one-paragraph explanation of cursor-based pagination limits and rate-limit headers (`X-RateLimit-Remaining`, `Retry-After`) in a REST API.", True),
        ("single_para", "Write a brief documentation paragraph explaining how content-addressed artifact caches avoid rebuilding unchanged packages in CI.", False),
        ("single_para", "Draft a concise explanation of how circuit breakers stop cascading failures when an upstream recommendation service slows down.", False),
        ("single_para", "Write a one-paragraph guide note explaining why structured JSON logs with a `request_id` field simplify distributed tracing.", True),
        ("single_para", "Draft a short release note explaining how schema registry compatibility modes (`BACKWARD` vs `FULL`) protect downstream Kafka consumers.", True),
        ("single_para", "Write a one-paragraph explanation of how blue-green deployments reduce rollback time compared to in-place instance updates.", False),
        ("single_para", "Draft a concise runbook note describing how to verify DNS TTL propagation before switching traffic to a new load balancer.", False),
        ("multi_para_markdown", "Write a two-paragraph runbook section with a fenced Bash command for diagnosing high disk usage caused by unrotated container logs in `/var/lib/docker`.", True),
        ("multi_para_markdown", "Write a short RFC motivation section (two paragraphs and a 3-bullet summary) proposing per-tenant rate limits at the API gateway.", False),
        ("multi_para_markdown", "Write a two-paragraph quickstart guide with a fenced `curl` example showing how to authenticate with a bearer token and fetch `/v1/events`.", True),
        ("multi_para_markdown", "Write a two-paragraph postmortem summary describing a 14-minute outage caused by a misconfigured Redis eviction policy (`noeviction` instead of `allkeys-lru`).", True),
        ("multi_para_markdown", "Write a short migration guide (two paragraphs plus a fenced TOML config example) for upgrading `config/worker.toml` from schema v1 to v2.", True),
        ("multi_para_markdown", "Write a two-paragraph documentation section explaining how webhook retry schedules work, including a brief bulleted list of backoff intervals.", False),
        ("multi_para_markdown", "Write a concise design-doc excerpt (two paragraphs) comparing pull-based metrics scraping against push-based telemetry for short-lived batch jobs.", False),
        ("multi_para_markdown", "Write a two-paragraph operational guide with a fenced SQL check for detecting PostgreSQL table bloat before scheduling `VACUUM` maintenance.", True),
        ("multi_para_markdown", "Write a short onboarding doc section (two paragraphs and a fenced command) explaining how to seed a local development database with `scripts/seed_dev.py`.", True),
        ("multi_para_markdown", "Write a two-paragraph architecture note explaining how dead-letter queues isolate poison messages without blocking healthy partition consumers.", False),
        ("multi_para_markdown", "Write a concise documentation section (two paragraphs plus a small JSON example) explaining how feature flag targeting rules evaluate percentage rollouts.", True),
        ("multi_para_markdown", "Write a two-paragraph security review checklist note covering secret rotation, least-privilege IAM roles, and audit logging for CI runners.", False),
        ("multi_para_markdown", "Write a short performance tuning note (two paragraphs) explaining how HTTP/2 connection multiplexing changes load-balancer stickiness.", False),
        ("multi_para_markdown", "Write a two-paragraph disaster recovery guide explaining the difference between RPO and RTO with a concrete backup-restore example.", False),
        ("multi_para_markdown", "Write a short developer handbook section (two paragraphs and a bulleted list) on writing reproducible bug reports with minimal test cases.", False),
    ]

    essay_prompts: list[tuple[str, str, bool]] = [
        ("single_para", "Write a single analytical paragraph on why mid-sized cities struggle to fund bus rapid transit even when ridership projections are strong.", False),
        ("single_para", "Write one thoughtful paragraph explaining why scientific journals face tension between publishing novel results and publishing replication studies.", False),
        ("single_para", "Write a concise paragraph discussing how right-to-repair laws affect both independent repair shops and electronics manufacturers.", False),
        ("single_para", "Write one analytical paragraph on why public libraries remain heavily used even in neighborhoods with high broadband adoption.", False),
        ("single_para", "Write a single paragraph explaining how grid-scale battery storage changes the economics of daytime solar generation.", False),
        ("single_para", "Write one paragraph examining why open-source maintainers often experience burnout on critical infrastructure projects.", False),
        ("single_para", "Write a concise paragraph on how congestion pricing in city centers alters delivery logistics as well as commuter behavior.", False),
        ("single_para", "Write one paragraph analyzing why rural hospitals face different staffing pressures than urban medical centers.", False),
        ("single_para", "Write a single analytical paragraph on the trade-offs between centralized water reservoirs and distributed stormwater capture in dry regions.", False),
        ("single_para", "Write one paragraph explaining why standardized shipping containers reshaped coastal port geography in the twentieth century.", False),
        ("single_para", "Write a concise paragraph discussing how local zoning rules influence the availability of neighborhood childcare centers.", False),
        ("single_para", "Write one paragraph examining why digital preservation of news websites is harder than archiving printed newspapers.", False),
        ("single_para", "Write a single paragraph on how vocational apprenticeships help bridge regional labor shortages in electrical and HVAC trades.", False),
        ("single_para", "Write one analytical paragraph explaining why small independent bookstores have adapted more effectively to e-commerce than mid-sized chains.", False),
        ("single_para", "Write a concise paragraph on the unintended effects of strict standardized testing on high school science curricula.", False),
        ("multi_para_markdown", "Write a two-paragraph essay excerpt analyzing why urban tree canopies reduce summer heat islands and why low-income neighborhoods often have fewer trees.", False),
        ("multi_para_markdown", "Write two analytical paragraphs examining how night trains compete with short-haul regional flights across Europe and what booking barriers remain.", False),
        ("multi_para_markdown", "Write a two-paragraph commentary on why municipal broadband initiatives succeed in some small towns while stalling in others.", False),
        ("multi_para_markdown", "Write two paragraphs discussing how precision drip irrigation interacts with historical water-rights rules in arid river basins.", False),
        ("multi_para_markdown", "Write a two-paragraph analysis of why building retrofits for heat pumps and insulation lag behind new-construction energy codes.", False),
        ("multi_para_markdown", "Write two paragraphs exploring how community land trusts keep housing affordable over multiple resale cycles.", False),
        ("multi_para_markdown", "Write a two-paragraph essay excerpt on the pros and cons of automated port cranes for maritime supply chain resilience and dockworker retraining.", False),
        ("multi_para_markdown", "Write two analytical paragraphs explaining how university technology-transfer offices balance open academic research with patent licensing.", False),
        ("multi_para_markdown", "Write a two-paragraph reflection on why pedestrian-friendly main streets often see higher retail foot traffic after removing curb parking.", False),
        ("multi_para_markdown", "Write two paragraphs examining how regional food hubs help small farms sell to school districts and hospitals.", False),
        ("multi_para_markdown", "Write a two-paragraph analysis of why flood-insurance pricing models struggle to keep pace with shifting coastal storm risks.", False),
        ("multi_para_markdown", "Write two paragraphs discussing how open museum image archives expand historical scholarship and classroom teaching.", False),
        ("multi_para_markdown", "Write a two-paragraph essay excerpt on why pharmaceutical generic competition depends on regulatory review speed as much as patent expiration.", False),
        ("multi_para_markdown", "Write two analytical paragraphs on how remote work has changed daytime transit demand and downtown commercial tax bases.", False),
        ("multi_para_markdown", "Write a two-paragraph commentary on why clear plain-language public health guidance builds more trust than overly technical press releases.", False),
    ]

    workplace_chat_prompts: list[tuple[str, str, bool]] = [
        ("single_para", "Write a short Slack message updating a product team that the CSV export fix is live in staging and asking QA to verify large files before Thursday's release.", False),
        ("single_para", "Write a concise workplace chat reply explaining why we should delay deprecating the legacy v1 webhook endpoint by two weeks for three enterprise accounts.", False),
        ("single_para", "Draft a one-paragraph message to an engineering channel flagging that CI runners are queued up because of a broken Docker cache layer.", True),
        ("single_para", "Write a brief Slack update summarizing two decisions from the morning design review about simplifying the account recovery screen.", False),
        ("single_para", "Draft a short message asking a teammate to take a look at a pull request in `src/billing/invoices.py` before end of day.", True),
        ("single_para", "Write a concise Slack handoff note from an outgoing on-call engineer explaining that alert volume dropped after rolling back the search index change.", False),
        ("single_para", "Draft a one-paragraph message to finance and ops proposing a simpler approval threshold for routine security tool renewals under $25k.", False),
        ("single_para", "Write a short team chat note reminding everyone to freeze non-urgent schema migrations during Friday's billing run.", False),
        ("single_para", "Draft a concise reply to a customer success manager explaining what is causing intermittent PDF export timeouts and when the patch ships.", False),
        ("single_para", "Write a brief message proposing that we trim the weekly status meeting from 60 minutes to 30 minutes and move routine updates into a shared doc.", False),
        ("single_para", "Draft a one-paragraph Slack note flagging that two localization keys overflow the mobile checkout button in German.", False),
        ("single_para", "Write a short message thanking a teammate for debugging a flaky integration test and noting that `tests/test_sync.py` is green again.", True),
        ("single_para", "Draft a concise workplace chat update explaining why we paused the new onboarding email sequence after seeing higher unsubscribe rates.", False),
        ("single_para", "Write a brief message asking legal whether our updated data-retention policy needs customer notice thirty days in advance.", False),
        ("single_para", "Draft a short Slack message sharing that the staging database refresh finished early and test credentials are ready.", False),
        ("multi_para_markdown", "Write a two-paragraph Slack update with a short 3-bullet action list summarizing a cross-team kickoff for the Q3 analytics migration.", False),
        ("multi_para_markdown", "Write a concise two-paragraph incident update for an internal `#incidents` channel explaining what broke in checkout, what mitigated it, and next steps.", False),
        ("multi_para_markdown", "Write a two-paragraph workplace message giving constructive feedback on a draft customer FAQ and suggesting two small clarifications.", False),
        ("multi_para_markdown", "Write a short two-paragraph sprint retrospective note highlighting what went well with automated deploys and two bottlenecks to fix next sprint.", False),
        ("multi_para_markdown", "Write a two-paragraph internal announcement explaining a new PR size guideline and linking to the updated checklist in `CONTRIBUTING.md`.", True),
        ("multi_para_markdown", "Write a concise two-paragraph Slack handoff between regional support shifts summarizing three open high-priority customer tickets.", False),
        ("multi_para_markdown", "Write a two-paragraph message to engineering leadership proposing a dedicated reliability week to pay down flaky test suites and alerting noise.", False),
        ("multi_para_markdown", "Write a short two-paragraph update to sales and account managers clarifying which beta reporting features are ready for customer demos this quarter.", False),
        ("multi_para_markdown", "Write a two-paragraph workplace chat post summarizing findings from a 10-user usability study on the new permissions settings page.", False),
        ("multi_para_markdown", "Write a concise two-paragraph note to the data team flagging a discrepancy between the daily revenue dashboard and Stripe settlement totals.", False),
        ("multi_para_markdown", "Write a two-paragraph onboarding message welcoming a new backend engineer and listing three helpful starter links and setup steps.", False),
        ("multi_para_markdown", "Write a short two-paragraph message proposing how to triage urgent bug reports separately from general feature questions in `#support-eng`.", False),
        ("multi_para_markdown", "Write a two-paragraph update after a vendor security review outlining two minor configuration changes needed before contract sign-off.", False),
        ("multi_para_markdown", "Write a concise two-paragraph Slack post sharing the rollout plan for switching worker autoscaling from CPU usage to queue depth in `deploy/hpa.yaml`.", True),
        ("multi_para_markdown", "Write a two-paragraph message coordinating a dry run before next week's quarterly board presentation so finance and product numbers align.", False),
    ]

    domain_groups = [
        ("coding", coding_prompts),
        ("technical_docs", technical_docs_prompts),
        ("essay", essay_prompts),
        ("workplace_chat", workplace_chat_prompts),
    ]
    for domain, items in domain_groups:
        for idx, (style, prompt_text, expects_code) in enumerate(items, start=1):
            specs.append(
                PromptSpec(
                    prompt_id=f"{domain}_{idx:03d}",
                    domain=domain,
                    style=style,
                    prompt=prompt_text,
                    expects_code=expects_code,
                )
            )
    return tuple(specs)


PROMPT_BANK: tuple[PromptSpec, ...] = _build_prompt_bank()


def get_prompts(
    *,
    domains: Sequence[str] | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> list[PromptSpec]:
    """Return prompts from `PROMPT_BANK`, optionally filtered by domain and shuffled."""
    if domains:
        wanted = {d.strip().lower() for d in domains}
        prompts = [p for p in PROMPT_BANK if p.domain in wanted]
    else:
        prompts = list(PROMPT_BANK)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(prompts)
    if limit is not None:
        prompts = prompts[: max(limit, 0)]
    return prompts


# ---------------------------------------------------------------------------
# Out-of-lexicon detection against classical.py's verb frames
# ---------------------------------------------------------------------------

_CLASSICAL_LEXICON_LEMMAS: frozenset[str] = frozenset(
    set(DENOMINAL_VERB_FRAMES.keys()) | set(INANIMATE_VERB_FRAMES.keys())
)

# Irregular or special inflections for classical.py's verb lemmas
_EXTRA_INFLECTIONS: dict[str, tuple[str, ...]] = {
    "eat": ("eat", "eats", "ate", "eaten", "eating"),
    "bite": ("bite", "bites", "bit", "bitten", "biting"),
    "bleed": ("bleed", "bleeds", "bled", "bleeding"),
    "choke": ("choke", "chokes", "choked", "choking"),
    "starve": ("starve", "starves", "starved", "starving"),
    "refuse": ("refuse", "refuses", "refused", "refusing"),
    "sabotage": ("sabotage", "sabotages", "sabotaged", "sabotaging"),
    "stumble": ("stumble", "stumbles", "stumbled", "stumbling"),
    "wrestle": ("wrestle", "wrestles", "wrestled", "wrestling"),
    "punish": ("punish", "punishes", "punished", "punishing"),
    "footnote": ("footnote", "footnotes", "footnoted", "footnoting"),
    "gate": ("gate", "gates", "gated", "gating"),
    "surface": ("surface", "surfaces", "surfaced", "surfacing"),
    "sunset": ("sunset", "sunsets", "sunsetted", "sunsetting"),
    "triage": ("triage", "triages", "triaged", "triaging"),
    "bridge": ("bridge", "bridges", "bridged", "bridging"),
    "funnel": ("funnel", "funnels", "funneled", "funnelled", "funneling", "funnelling"),
    "silo": ("silo", "silos", "siloed", "siloing"),
    "leverage": ("leverage", "leverages", "leveraged", "leveraging"),
    "pipeline": ("pipeline", "pipelines", "pipelined", "pipelining"),
    "shelve": ("shelve", "shelves", "shelved", "shelving"),
    "table": ("table", "tables", "tabled", "tabling"),
    "sandbox": ("sandbox", "sandboxes", "sandboxed", "sandboxing"),
    "benchmark": ("benchmark", "benchmarks", "benchmarked", "benchmarking"),
    "bottleneck": ("bottleneck", "bottlenecks", "bottlenecked", "bottlenecking"),
}

# Bare lemmas that are also very common literal nouns when preceded by a determiner.
_NOUN_AMBIGUOUS_BASE_LEMMAS: frozenset[str] = frozenset(
    {
        "anchor",
        "architect",
        "benchmark",
        "bite",
        "bottleneck",
        "bridge",
        "bucket",
        "calendar",
        "champion",
        "demand",
        "eyeball",
        "fool",
        "footnote",
        "funnel",
        "gate",
        "hunger",
        "instrument",
        "leverage",
        "mirror",
        "pipeline",
        "sandbox",
        "scaffold",
        "scream",
        "shadow",
        "shoulder",
        "signpost",
        "silo",
        "spotlight",
        "sunset",
        "surface",
        "table",
        "triage",
        "trick",
        "whisper",
        "workshop",
    }
)

_DETERMINERS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "my",
        "our",
        "your",
        "his",
        "her",
        "its",
        "their",
        "each",
        "every",
        "any",
        "some",
        "no",
        "one",
        "another",
    }
)


def _build_lexicon_surface_map() -> dict[str, str]:
    surface_to_lemma: dict[str, str] = {}
    for lemma in _CLASSICAL_LEXICON_LEMMAS:
        forms: set[str] = {lemma}
        if lemma in _EXTRA_INFLECTIONS:
            forms.update(_EXTRA_INFLECTIONS[lemma])
        else:
            # Regular English verb inflections
            if lemma.endswith(("s", "x", "z", "ch", "sh")):
                forms.add(lemma + "es")
            elif lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
                forms.add(lemma[:-1] + "ies")
            else:
                forms.add(lemma + "s")
            if lemma.endswith("e"):
                forms.add(lemma + "d")
                forms.add(lemma[:-1] + "ing")
            elif lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
                forms.add(lemma[:-1] + "ied")
                forms.add(lemma + "ing")
            else:
                forms.add(lemma + "ed")
                forms.add(lemma + "ing")
        for form in forms:
            surface_to_lemma[form.lower()] = lemma

    for surface, lemma in DENOMINAL_SURFACE_TO_LEMMA.items():
        if lemma in _CLASSICAL_LEXICON_LEMMAS:
            surface_to_lemma[surface.lower()] = lemma
    return surface_to_lemma


_LEXICON_SURFACE_TO_LEMMA: dict[str, str] = _build_lexicon_surface_map()


def find_lexicon_verbs(text: str, *, nlp: Any | None = None) -> list[str]:
    """Return sorted classical lexicon verb lemmas found in prose (ignoring code spans).

    Checks against `classical.py`'s `DENOMINAL_VERB_FRAMES` and `INANIMATE_VERB_FRAMES`.
    If a spaCy pipeline is provided via `nlp`, uses POS tags (`VERB`) for exact disambiguation;
    otherwise uses deterministic inflected-form matching with determiner-noun guarding.
    """
    prose = strip_fenced_code_blocks(text)
    masked = protect_spans(prose).masked
    matched: set[str] = set()

    if nlp is not None:
        doc = nlp(masked)
        for tok in doc:
            if tok.pos_ != "VERB":
                continue
            low = tok.text.lower()
            lemma = DENOMINAL_SURFACE_TO_LEMMA.get(low, tok.lemma_.lower())
            if lemma in _CLASSICAL_LEXICON_LEMMAS:
                matched.add(lemma)
        return sorted(matched)

    words = [w for w in _WORD_RE.findall(masked.lower()) if not w.startswith("codeitem")]
    for idx, word in enumerate(words):
        lemma = _LEXICON_SURFACE_TO_LEMMA.get(word)
        if lemma is None:
            continue
        prev = words[idx - 1] if idx > 0 else ""
        # Skip obvious singular noun uses like "the table", "a gate", "this bridge".
        if word == lemma and lemma in _NOUN_AMBIGUOUS_BASE_LEMMAS and prev in _DETERMINERS:
            continue
        matched.add(lemma)
    return sorted(matched)


def is_out_of_lexicon(text: str, *, nlp: Any | None = None) -> bool:
    """Return True iff `text` contains zero verbs from `classical.py`'s verb frames."""
    return len(find_lexicon_verbs(text, nlp=nlp)) == 0


# ---------------------------------------------------------------------------
# Deduplication & OOD splitting
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize text for exact and n-gram deduplication."""
    no_fences = _FENCE_RE.sub(" ", text)
    words = _WORD_RE.findall(no_fences.lower())
    return " ".join(words)


def word_ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    """Extract word n-gram shingles from normalized text."""
    words = normalize_text(text).split()
    if not words:
        return set()
    eff_n = min(max(n, 1), len(words))
    return {tuple(words[i : i + eff_n]) for i in range(len(words) - eff_n + 1)}


def ngram_jaccard(text_a: str, text_b: str, n: int = 5) -> float:
    """Compute Jaccard similarity over word n-grams of two passages."""
    norm_a = normalize_text(text_a)
    norm_b = normalize_text(text_b)
    if not norm_a and not norm_b:
        return 1.0
    if norm_a == norm_b:
        return 1.0
    grams_a = word_ngrams(norm_a, n=n)
    grams_b = word_ngrams(norm_b, n=n)
    if not grams_a or not grams_b:
        return 0.0
    inter = len(grams_a & grams_b)
    union = len(grams_a | grams_b)
    return inter / union if union else 0.0


def deduplicate_samples(
    samples: Sequence[DatasetSample],
    *,
    similarity_threshold: float = 0.55,
    ngram_n: int = 5,
) -> tuple[list[DatasetSample], int]:
    """Remove exact and near-duplicate samples using normalized text + n-gram Jaccard.

    Returns `(kept_samples, num_dropped)`.
    """
    kept: list[DatasetSample] = []
    seen_norms: set[str] = set()
    kept_grams: list[set[tuple[str, ...]]] = []
    inverted_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    dropped = 0

    for sample in samples:
        norm = normalize_text(sample.text)
        if not norm or norm in seen_norms:
            dropped += 1
            continue

        grams = word_ngrams(norm, n=ngram_n)
        candidate_indices: set[int] = set()
        for g in grams:
            if g in inverted_index:
                candidate_indices.update(inverted_index[g])

        is_near_dup = False
        for idx in candidate_indices:
            other_grams = kept_grams[idx]
            inter = len(grams & other_grams)
            union = len(grams | other_grams)
            if union > 0 and (inter / union) >= similarity_threshold:
                is_near_dup = True
                break

        if is_near_dup:
            dropped += 1
            continue

        new_idx = len(kept)
        kept.append(sample)
        seen_norms.add(norm)
        kept_grams.append(grams)
        for g in grams:
            inverted_index[g].append(new_idx)

    return kept, dropped


def _verify_no_cross_split_leakage(
    splits: Mapping[str, Sequence[DatasetSample]],
    *,
    similarity_threshold: float,
    ngram_n: int,
) -> None:
    split_names = list(splits.keys())
    norms_by_split: dict[str, set[str]] = {
        name: {normalize_text(s.text) for s in items} for name, items in splits.items()
    }
    grams_by_split: dict[str, list[set[tuple[str, ...]]]] = {
        name: [word_ngrams(s.text, n=ngram_n) for s in items] for name, items in splits.items()
    }
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = norms_by_split[a] & norms_by_split[b]
            if overlap:
                raise RuntimeError(f"Exact duplicate leaked between {a} and {b}")
            for ga in grams_by_split[a]:
                for gb in grams_by_split[b]:
                    union = len(ga | gb)
                    if union > 0 and (len(ga & gb) / union) >= similarity_threshold:
                        raise RuntimeError(f"Near-duplicate leaked between {a} and {b}")


def build_splits(
    samples: Sequence[DatasetSample],
    *,
    output_dir: str | Path | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.20,
    similarity_threshold: float = 0.55,
    ngram_n: int = 5,
    seed: int = 42,
    group_by_prompt: bool = True,
    prefer_ood_in_test: bool = True,
    nlp: Any | None = None,
) -> dict[str, list[DatasetSample]]:
    """Deduplicate samples, tag `is_out_of_lexicon`, and split into `train`, `val`, `test_ood`.

    Guarantees zero exact or near-duplicate leakage across splits (and zero prompt-id
    leakage when `group_by_prompt=True`).
    """
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0 and val_ratio + test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0, 1) with val_ratio + test_ratio < 1.")

    deduped, dropped_count = deduplicate_samples(
        samples,
        similarity_threshold=similarity_threshold,
        ngram_n=ngram_n,
    )

    # Tag every sample with out-of-lexicon status.
    tagged: list[DatasetSample] = []
    for sample in deduped:
        ool = (
            sample.is_out_of_lexicon
            if sample.is_out_of_lexicon is not None
            else is_out_of_lexicon(sample.text, nlp=nlp)
        )
        tagged.append(replace(sample, is_out_of_lexicon=ool))

    # Group by prompt_id (if available) to prevent multi-model responses to the same prompt
    # from leaking across splits.
    groups_map: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in tagged:
        key = sample.prompt_id if (group_by_prompt and sample.prompt_id) else sample.sample_id
        groups_map[key].append(sample)

    # Stratify groups by primary domain so each split maintains domain diversity.
    groups_by_domain: dict[str, list[list[DatasetSample]]] = defaultdict(list)
    for group in groups_map.values():
        groups_by_domain[group[0].domain].append(group)

    rng = random.Random(seed)
    train_items: list[DatasetSample] = []
    val_items: list[DatasetSample] = []
    test_items: list[DatasetSample] = []

    for domain in sorted(groups_by_domain.keys()):
        dom_groups = groups_by_domain[domain]
        rng.shuffle(dom_groups)

        if prefer_ood_in_test:
            # Sort groups with higher out-of-lexicon fraction first for test_ood selection.
            dom_groups.sort(
                key=lambda grp: sum(1 for s in grp if s.is_out_of_lexicon) / max(len(grp), 1),
                reverse=True,
            )

        n_groups = len(dom_groups)
        n_test = round(n_groups * test_ratio)
        if test_ratio > 0 and n_test == 0 and n_groups >= 3:
            n_test = 1
        n_val = round(n_groups * val_ratio)
        if val_ratio > 0 and n_val == 0 and (n_groups - n_test) >= 2:
            n_val = 1
        if n_test + n_val >= n_groups and n_groups >= 1:
            # Always keep at least one group in train when possible.
            if n_groups == 1:
                n_test, n_val = 0, 0
            elif n_groups == 2:
                n_test, n_val = (1, 0) if test_ratio >= val_ratio else (0, 1)
            else:
                n_val = max(0, min(n_val, n_groups - n_test - 1))

        chosen_test = dom_groups[:n_test]
        remaining = dom_groups[n_test:]
        rng.shuffle(remaining)
        chosen_val = remaining[:n_val]
        chosen_train = remaining[n_val:]

        for grp in chosen_test:
            test_items.extend(replace(s, split="test_ood") for s in grp)
        for grp in chosen_val:
            val_items.extend(replace(s, split="val") for s in grp)
        for grp in chosen_train:
            train_items.extend(replace(s, split="train") for s in grp)

    splits = {"train": train_items, "val": val_items, "test_ood": test_items}
    _verify_no_cross_split_leakage(
        splits,
        similarity_threshold=similarity_threshold,
        ngram_n=ngram_n,
    )

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for split_name, items in splits.items():
            save_jsonl(out_path / f"{split_name}.jsonl", items, append=False)
        stats = {
            "total_input": len(samples),
            "dedup_dropped": dropped_count,
            "total_kept": len(tagged),
            "splits": {
                name: {
                    "count": len(items),
                    "out_of_lexicon_count": sum(1 for s in items if s.is_out_of_lexicon),
                    "out_of_lexicon_rate": (
                        round(sum(1 for s in items if s.is_out_of_lexicon) / len(items), 4)
                        if items
                        else 0.0
                    ),
                }
                for name, items in splits.items()
            },
        }
        (out_path / "split_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return splits


# ---------------------------------------------------------------------------
# Cost tracking & resilient OpenRouter / OpenAI httpx client
# ---------------------------------------------------------------------------


class BudgetExceededError(RuntimeError):
    """Raised when a configured USD cost cap or request cap is reached."""


class CostTracker:
    """Thread-safe tracker enforcing `--max-cost-usd` and `--max-requests`."""

    def __init__(
        self,
        *,
        max_cost_usd: float | None = None,
        max_requests: int | None = None,
        pricing_per_1m: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.max_cost_usd = max_cost_usd
        self.max_requests = max_requests
        self._pricing = dict(MODEL_PRICING_PER_1M)
        if pricing_per_1m:
            self._pricing.update(pricing_per_1m)
        self._lock = threading.Lock()
        self.requests: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.by_model: dict[str, dict[str, float | int]] = {}

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        in_rate, out_rate = self._pricing.get(model, DEFAULT_PRICING_PER_1M)
        return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0

    def check_budget(self) -> None:
        with self._lock:
            if self.max_requests is not None and self.requests >= self.max_requests:
                raise BudgetExceededError(
                    f"Request cap reached ({self.requests}/{self.max_requests})."
                )
            if self.max_cost_usd is not None and self.total_cost_usd >= self.max_cost_usd:
                raise BudgetExceededError(
                    f"Cost cap reached (${self.total_cost_usd:.4f} / ${self.max_cost_usd:.4f})."
                )

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        cost = self.estimate_cost(model, prompt_tokens, completion_tokens)
        with self._lock:
            self.requests += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_cost_usd += cost
            entry = self.by_model.setdefault(
                model,
                {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
            )
            entry["requests"] = int(entry["requests"]) + 1
            entry["prompt_tokens"] = int(entry["prompt_tokens"]) + prompt_tokens
            entry["completion_tokens"] = int(entry["completion_tokens"]) + completion_tokens
            entry["cost_usd"] = float(entry["cost_usd"]) + cost
        return cost

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": self.requests,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "max_cost_usd": self.max_cost_usd,
                "max_requests": self.max_requests,
                "by_model": {k: dict(v) for k, v in self.by_model.items()},
            }


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class LLMClient:
    """Thread-safe OpenRouter/OpenAI chat-completions client using `httpx` with backoff."""

    def __init__(
        self,
        *,
        provider: str = "auto",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 5,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 30.0,
        cost_tracker: CostTracker | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider.strip().lower()
        self._explicit_api_key = api_key
        self._explicit_base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.cost_tracker = cost_tracker or CostTracker()
        self._sleep_fn = sleep_fn
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds)),
            transport=transport,
        )

    def __repr__(self) -> str:
        return (
            f"LLMClient(provider={self.provider!r}, timeout_seconds={self.timeout_seconds}, "
            f"max_retries={self.max_retries})"
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _resolve_endpoint(self, model: str) -> tuple[str, str, str, str]:
        """Return `(resolved_provider, url, api_key, wire_model)` without ever logging keys."""
        prov = self.provider
        if prov == "auto":
            if "/" in model and os.environ.get("OPENROUTER_API_KEY"):
                prov = "openrouter"
            elif os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
                prov = "openrouter"
            elif os.environ.get("OPENAI_API_KEY"):
                prov = "openai"
            elif os.environ.get("OPENROUTER_API_KEY"):
                prov = "openrouter"
            elif self._explicit_api_key:
                prov = "openrouter" if "/" in model else "openai"
            else:
                raise RuntimeError(
                    "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in the environment."
                )

        if prov == "openrouter":
            key = self._explicit_api_key or os.environ.get("OPENROUTER_API_KEY", "")
            url = self._explicit_base_url or OPENROUTER_BASE_URL
            wire_model = model
        elif prov == "openai":
            key = self._explicit_api_key or os.environ.get("OPENAI_API_KEY", "")
            url = self._explicit_base_url or OPENAI_BASE_URL
            wire_model = model.split("/", 1)[1] if model.startswith("openai/") else model
        else:
            raise ValueError(f"Unsupported provider {prov!r}. Use 'auto', 'openrouter', or 'openai'.")

        if not key:
            env_name = "OPENROUTER_API_KEY" if prov == "openrouter" else "OPENAI_API_KEY"
            raise RuntimeError(f"Missing API key for provider {prov!r} (expected {env_name}).")
        return prov, url, key, wire_model

    def _retry_delay(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(max(float(retry_after), 0.0), self.backoff_max_seconds)
                except ValueError:
                    pass
        base = min(self.backoff_base_seconds * (2 ** attempt), self.backoff_max_seconds)
        jitter = random.uniform(0.0, base * 0.25)
        return base + jitter

    def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request with exponential backoff and budget tracking."""
        self.cost_tracker.check_budget()
        prov, url, api_key, wire_model = self._resolve_endpoint(model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if prov == "openrouter":
            headers["X-Title"] = "psycholinguistic-reversal"

        payload: dict[str, Any] = {
            "model": wire_model,
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = dict(response_format)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(url, headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Network error calling {prov} for model {model} after {attempt + 1} attempts: "
                        f"{type(exc).__name__}"
                    ) from exc
                self._sleep_fn(self._retry_delay(attempt, None))
                continue

            if resp.status_code in _RETRYABLE_STATUS_CODES:
                last_error = RuntimeError(f"HTTP {resp.status_code} from {prov} for model {model}")
                if attempt >= self.max_retries:
                    raise last_error
                self._sleep_fn(self._retry_delay(attempt, resp))
                continue

            if resp.status_code >= 400:
                # Never echo request headers or API keys in error messages.
                snippet = resp.text[:200].replace("\n", " ")
                raise RuntimeError(
                    f"API request failed with HTTP {resp.status_code} ({prov}, model={model}): {snippet}"
                )

            data = resp.json()
            choices = data.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                raise RuntimeError(f"Malformed completion response from {prov} for model {model}")
            content = str((choices[0].get("message") or {}).get("content") or "").strip()
            usage = data.get("usage") or {}
            prompt_tokens = int(
                usage.get("prompt_tokens")
                or sum(len(str(m.get("content", "")).split()) for m in messages) * 2
            )
            completion_tokens = int(
                usage.get("completion_tokens") or max(len(content.split()) * 2, 1)
            )
            cost = self.cost_tracker.record(model, prompt_tokens, completion_tokens)
            return LLMResponse(
                content=content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
            )

        raise RuntimeError(f"Failed calling {prov} for model {model}: {last_error}")


# ---------------------------------------------------------------------------
# Stage 1: Real LLM prose generation
# ---------------------------------------------------------------------------

_GENERATION_SYSTEM_PROMPT = (
    "You are a knowledgeable writer and engineer responding naturally to everyday user requests. "
    "Write clear, authentic prose matching the requested format and length. "
    "Output only the requested passage with no preamble or meta-commentary."
)


def _slugify_model(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


def generate_real_corpus(
    *,
    output_dir: str | Path = DEFAULT_DATASET_DIR,
    models: Sequence[str] = DEFAULT_GENERATION_MODELS,
    domains: Sequence[str] | None = None,
    samples_per_model: int | None = None,
    resume: bool = True,
    max_workers: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 700,
    seed: int = 42,
    llm_client: LLMClient | None = None,
    provider: str = "auto",
    max_cost_usd: float | None = None,
    max_requests: int | None = None,
    timeout_seconds: float = 60.0,
) -> list[DatasetSample]:
    """Generate authentic uncurated LLM prose across real models and 4 domains with JSONL resume."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    corpus_file = out_path / "raw_corpus.jsonl"

    existing_samples: list[DatasetSample] = []
    completed_ids: set[str] = set()
    if resume and corpus_file.exists():
        for row in load_jsonl(corpus_file, skip_corrupt=True):
            sample = DatasetSample.from_dict(row)
            if sample.sample_id not in completed_ids:
                existing_samples.append(sample)
                completed_ids.add(sample.sample_id)

    prompts = get_prompts(domains=domains, limit=samples_per_model, seed=seed)
    jobs: list[tuple[str, PromptSpec, str]] = []
    for model in models:
        slug = _slugify_model(model)
        for spec in prompts:
            sample_id = f"{spec.prompt_id}__{slug}"
            if sample_id in completed_ids:
                continue
            jobs.append((model, spec, sample_id))

    if not jobs:
        return existing_samples

    owns_client = llm_client is None
    tracker = llm_client.cost_tracker if llm_client is not None else CostTracker(
        max_cost_usd=max_cost_usd,
        max_requests=max_requests,
    )
    client = llm_client or LLMClient(
        provider=provider,
        timeout_seconds=timeout_seconds,
        cost_tracker=tracker,
    )

    write_lock = threading.Lock()
    stop_event = threading.Event()
    newly_generated: list[DatasetSample] = []

    def _run_one(model: str, spec: PromptSpec, sample_id: str) -> DatasetSample | None:
        if stop_event.is_set():
            return None
        messages = [
            {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": spec.prompt},
        ]
        try:
            resp = client.complete(
                model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except BudgetExceededError:
            stop_event.set()
            return None

        if not resp.content:
            return None

        sample = DatasetSample(
            sample_id=sample_id,
            domain=spec.domain,
            text=resp.content,
            source_model=model,
            prompt_id=spec.prompt_id,
            split="",
            is_out_of_lexicon=is_out_of_lexicon(resp.content),
            is_multipara="\n\n" in resp.content,
            metadata={
                "style": spec.style,
                "expects_code": spec.expects_code,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "cost_usd": round(resp.cost_usd, 6),
            },
        )
        with write_lock:
            save_jsonl(corpus_file, [sample], append=True)
            newly_generated.append(sample)
        return sample

    try:
        if max_workers <= 1:
            for model, spec, sample_id in jobs:
                if stop_event.is_set():
                    break
                _run_one(model, spec, sample_id)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(_run_one, model, spec, sample_id)
                    for model, spec, sample_id in jobs
                ]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
    finally:
        if owns_client:
            client.close()

    return existing_samples + newly_generated


# ---------------------------------------------------------------------------
# Stage 3: Strong-teacher distillation & hard-negative mining
# ---------------------------------------------------------------------------

# Clean, non-leaked editorial prompt (never names psycholinguistic features or lexicons).
_TEACHER_SYSTEM_PROMPT = (
    "You are a senior prose and technical editor. Given an input passage, produce a JSON object "
    "with four keys:\n"
    '1. "needs_edit": boolean — true if the passage can be made clearer, more natural, and easier '
    "to read; false if it is already crisp, natural, and direct.\n"
    '2. "chosen": string — if needs_edit is true, rewrite the passage so it reads like clear, '
    "direct, natural human prose while preserving the exact meaning, technical accuracy, tone, "
    "markdown structure, and EVERY inline code span, fenced code block, filename, identifier, and "
    "URL verbatim. If needs_edit is false, return the input passage verbatim.\n"
    '3. "over_edited": string — a flawed alternative rewrite that over-edits the passage, drifts '
    "in meaning, drops nuance, or alters tone unnecessarily.\n"
    '4. "clunky_literal": string — a flawed alternative rewrite that is stiff, wordy, mechanical, '
    "or awkward to read.\n"
    "Return ONLY valid JSON with those four keys and no markdown commentary outside the JSON."
)

_IDENTITY_TEACHER_SYSTEM_PROMPT = (
    "You are a senior prose editor creating contrast examples for a rewriting model. "
    "The input passage is already clear and natural and should NOT be changed. "
    'Return ONLY a valid JSON object with one key, "over_edited", containing a flawed '
    "over-edited rewrite that unnecessarily rewords, bloats, or slightly distorts the passage "
    "while keeping any code spans intact."
)


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, tolerating optional markdown code fences."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Teacher response did not decode to a JSON object.")
    return parsed


def distill_training_pairs(
    samples: Sequence[DatasetSample],
    *,
    output_dir: str | Path = DEFAULT_DATASET_DIR,
    output_filename: str = "distilled_pairs.jsonl",
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    identity_samples: Sequence[DatasetSample] | None = None,
    resume: bool = True,
    max_workers: int = 4,
    temperature: float = 0.2,
    max_tokens: int = 1600,
    min_code_preservation: float = 1.0,
    llm_client: LLMClient | None = None,
    provider: str = "auto",
    max_cost_usd: float | None = None,
    max_requests: int | None = None,
    timeout_seconds: float = 90.0,
) -> list[DistilledPair]:
    """Distill teacher rewrites, hard negatives, and identity/no-op pairs with JSONL resume."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pairs_file = out_path / output_filename

    existing_pairs: list[DistilledPair] = []
    completed_ids: set[str] = set()
    if resume and pairs_file.exists():
        for row in load_jsonl(pairs_file, skip_corrupt=True):
            pair = DistilledPair.from_dict(row)
            if pair.pair_id not in completed_ids:
                existing_pairs.append(pair)
                completed_ids.add(pair.pair_id)

    # Combine regular samples and explicit identity/no-op samples.
    work_items: list[tuple[DatasetSample, bool]] = []
    for sample in samples:
        is_id = bool(sample.metadata.get("is_identity")) or sample.domain == "human_control"
        pair_id = f"pair_{sample.sample_id}"
        if pair_id not in completed_ids:
            work_items.append((sample, is_id))
    for sample in identity_samples or ():
        pair_id = f"pair_{sample.sample_id}"
        if pair_id not in completed_ids:
            work_items.append((sample, True))

    if not work_items:
        return existing_pairs

    owns_client = llm_client is None
    tracker = llm_client.cost_tracker if llm_client is not None else CostTracker(
        max_cost_usd=max_cost_usd,
        max_requests=max_requests,
    )
    client = llm_client or LLMClient(
        provider=provider,
        timeout_seconds=timeout_seconds,
        cost_tracker=tracker,
    )

    write_lock = threading.Lock()
    stop_event = threading.Event()
    new_pairs: list[DistilledPair] = []

    def _distill_one(sample: DatasetSample, force_identity: bool) -> DistilledPair | None:
        if stop_event.is_set():
            return None
        pair_id = f"pair_{sample.sample_id}"
        sys_prompt = _IDENTITY_TEACHER_SYSTEM_PROMPT if force_identity else _TEACHER_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": sample.text},
        ]
        max_parse_retries = 2
        resp: LLMResponse | None = None
        parsed: dict[str, Any] | None = None
        last_parse_err: Exception | None = None
        sleep_fn = getattr(client, "_sleep_fn", time.sleep)
        for attempt in range(max_parse_retries + 1):
            if stop_event.is_set():
                return None
            try:
                resp = client.complete(
                    teacher_model,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            except BudgetExceededError:
                stop_event.set()
                return None

            try:
                parsed = _extract_json_object(resp.content)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                last_parse_err = exc
                if attempt < max_parse_retries:
                    delay = min(0.5 * (2**attempt), 2.0)
                    logger.warning(
                        "Failed to parse JSON response for %s (attempt %d/%d): %s; retrying in %.2fs",
                        sample.sample_id,
                        attempt + 1,
                        max_parse_retries + 1,
                        exc,
                        delay,
                    )
                    sleep_fn(delay)

        if parsed is None or resp is None:
            logger.warning(
                "Skipping %s after %d unparseable JSON responses: %s",
                sample.sample_id,
                max_parse_retries + 1,
                last_parse_err,
            )
            return None
        over_edited = str(parsed.get("over_edited") or "").strip()
        clunky_literal = str(parsed.get("clunky_literal") or "").strip()

        if force_identity:
            is_identity = True
            chosen = sample.text
        else:
            needs_edit = bool(parsed.get("needs_edit", True))
            raw_chosen = str(parsed.get("chosen") or "").strip()
            if not needs_edit or not raw_chosen or normalize_text(raw_chosen) == normalize_text(sample.text):
                is_identity = True
                chosen = sample.text
            else:
                is_identity = False
                chosen = raw_chosen

        cpr = code_preservation_rate(sample.text, chosen)
        if not is_identity and cpr < min_code_preservation:
            # Skip corrupted rewrites that dropped protected code spans rather than poisoning training.
            return None

        hard_negatives: list[dict[str, str]] = []
        seen_neg_norms: set[str] = {normalize_text(chosen)}
        if over_edited and normalize_text(over_edited) not in seen_neg_norms:
            hard_negatives.append({"type": "over_edited", "text": over_edited})
            seen_neg_norms.add(normalize_text(over_edited))
        if clunky_literal and normalize_text(clunky_literal) not in seen_neg_norms:
            hard_negatives.append({"type": "clunky_literal", "text": clunky_literal})
            seen_neg_norms.add(normalize_text(clunky_literal))
        if not is_identity and normalize_text(sample.text) not in seen_neg_norms:
            hard_negatives.append({"type": "dense_original", "text": sample.text})
            seen_neg_norms.add(normalize_text(sample.text))

        primary_rejected = hard_negatives[0]["text"] if hard_negatives else ""

        pair = DistilledPair(
            pair_id=pair_id,
            sample_id=sample.sample_id,
            domain=sample.domain,
            source=sample.text,
            chosen=chosen,
            rejected=primary_rejected,
            hard_negatives=hard_negatives,
            is_identity=is_identity,
            teacher_model=teacher_model,
            source_model=sample.source_model,
            split=sample.split or "train",
            is_out_of_lexicon=sample.is_out_of_lexicon,
            code_preservation_rate=cpr,
            metadata={
                "prompt_id": sample.prompt_id,
                "teacher_cost_usd": round(resp.cost_usd, 6),
                "teacher_prompt_tokens": resp.prompt_tokens,
                "teacher_completion_tokens": resp.completion_tokens,
            },
        )
        with write_lock:
            save_jsonl(pairs_file, [pair], append=True)
            new_pairs.append(pair)
        return pair

    try:
        if max_workers <= 1:
            for sample, force_id in work_items:
                if stop_event.is_set():
                    break
                _distill_one(sample, force_id)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(_distill_one, sample, force_id)
                    for sample, force_id in work_items
                ]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
    finally:
        if owns_client:
            client.close()

    return existing_pairs + new_pairs


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m psycho_rev.corpus_builder",
        description="Build resumable real-world LLM corpora, OOD splits, and teacher-distilled pairs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--output-dir",
            type=Path,
            default=DEFAULT_DATASET_DIR,
            help=f"Directory for JSONL checkpoints and splits (default: {DEFAULT_DATASET_DIR})",
        )
        p.add_argument(
            "--resume",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Resume from existing JSONL checkpoints (default: True)",
        )
        p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling/splitting")

    def _add_api_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--provider",
            choices=("auto", "openrouter", "openai"),
            default="auto",
            help="LLM API provider (default: auto)",
        )
        p.add_argument("--max-workers", type=int, default=4, help="Concurrent HTTP worker threads")
        p.add_argument("--max-cost-usd", type=float, default=None, help="Hard USD budget cap")
        p.add_argument("--max-requests", type=int, default=None, help="Hard request count cap")
        p.add_argument("--timeout", type=float, default=60.0, help="HTTP request timeout in seconds")

    # generate
    p_gen = sub.add_parser("generate", help="Generate real-world LLM prose across models and domains")
    _add_common(p_gen)
    _add_api_args(p_gen)
    p_gen.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_GENERATION_MODELS),
        help="Model IDs to prompt (e.g. openai/gpt-4o-mini anthropic/claude-3.5-haiku)",
    )
    p_gen.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to sample prompts from",
    )
    p_gen.add_argument(
        "--samples-per-model",
        type=int,
        default=None,
        help="Maximum prompts per model (default: all matching prompts in PROMPT_BANK)",
    )
    p_gen.add_argument("--temperature", type=float, default=0.7)
    p_gen.add_argument("--max-tokens", type=int, default=700)

    # split
    p_split = sub.add_parser("split", help="Deduplicate, tag out-of-lexicon, and build train/val/test_ood splits")
    _add_common(p_split)
    p_split.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Input raw corpus JSONL (defaults to <output-dir>/raw_corpus.jsonl)",
    )
    p_split.add_argument("--val-ratio", type=float, default=0.15)
    p_split.add_argument("--test-ratio", type=float, default=0.20)
    p_split.add_argument("--similarity-threshold", type=float, default=0.55)
    p_split.add_argument("--ngram-n", type=int, default=5)
    p_split.add_argument(
        "--group-by-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep responses to the same prompt_id in the same split",
    )
    p_split.add_argument(
        "--prefer-ood-in-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prioritize out-of-lexicon passages for test_ood",
    )

    # distill
    p_dist = sub.add_parser("distill", help="Distill teacher rewrites, hard negatives, and identity pairs")
    _add_common(p_dist)
    _add_api_args(p_dist)
    p_dist.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Input split JSONL to distill (defaults to <output-dir>/train.jsonl)",
    )
    p_dist.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    p_dist.add_argument("--output-filename", default="distilled_pairs.jsonl")
    p_dist.add_argument("--temperature", type=float, default=0.2)
    p_dist.add_argument("--max-tokens", type=int, default=1600)
    p_dist.add_argument("--min-code-preservation", type=float, default=1.0)

    # all (end-to-end)
    p_all = sub.add_parser("all", help="Run generate -> split -> distill end-to-end")
    _add_common(p_all)
    _add_api_args(p_all)
    p_all.add_argument("--models", nargs="+", default=list(DEFAULT_GENERATION_MODELS))
    p_all.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    p_all.add_argument("--samples-per-model", type=int, default=None)
    p_all.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    p_all.add_argument("--val-ratio", type=float, default=0.15)
    p_all.add_argument("--test-ratio", type=float, default=0.20)
    p_all.add_argument("--similarity-threshold", type=float, default=0.55)
    p_all.add_argument("--min-code-preservation", type=float, default=1.0)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.cmd == "generate":
        samples = generate_real_corpus(
            output_dir=args.output_dir,
            models=args.models,
            domains=args.domains,
            samples_per_model=args.samples_per_model,
            resume=args.resume,
            max_workers=args.max_workers,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            provider=args.provider,
            max_cost_usd=args.max_cost_usd,
            max_requests=args.max_requests,
            timeout_seconds=args.timeout,
        )
        print(json.dumps({"cmd": "generate", "total_samples": len(samples), "output_dir": str(args.output_dir)}))
        return

    if args.cmd == "split":
        in_file = args.input_file or (args.output_dir / "raw_corpus.jsonl")
        raw_samples = [DatasetSample.from_dict(r) for r in load_jsonl(in_file, skip_corrupt=True)]
        splits = build_splits(
            raw_samples,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            similarity_threshold=args.similarity_threshold,
            ngram_n=args.ngram_n,
            seed=args.seed,
            group_by_prompt=args.group_by_prompt,
            prefer_ood_in_test=args.prefer_ood_in_test,
        )
        print(
            json.dumps(
                {
                    "cmd": "split",
                    "counts": {k: len(v) for k, v in splits.items()},
                    "output_dir": str(args.output_dir),
                }
            )
        )
        return

    if args.cmd == "distill":
        in_file = args.input_file or (args.output_dir / "train.jsonl")
        train_samples = [DatasetSample.from_dict(r) for r in load_jsonl(in_file, skip_corrupt=True)]
        pairs = distill_training_pairs(
            train_samples,
            output_dir=args.output_dir,
            output_filename=args.output_filename,
            teacher_model=args.teacher_model,
            resume=args.resume,
            max_workers=args.max_workers,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            min_code_preservation=args.min_code_preservation,
            provider=args.provider,
            max_cost_usd=args.max_cost_usd,
            max_requests=args.max_requests,
            timeout_seconds=args.timeout,
        )
        print(json.dumps({"cmd": "distill", "total_pairs": len(pairs), "output_dir": str(args.output_dir)}))
        return

    if args.cmd == "all":
        tracker = CostTracker(max_cost_usd=args.max_cost_usd, max_requests=args.max_requests)
        with LLMClient(
            provider=args.provider,
            timeout_seconds=args.timeout,
            cost_tracker=tracker,
        ) as client:
            samples = generate_real_corpus(
                output_dir=args.output_dir,
                models=args.models,
                domains=args.domains,
                samples_per_model=args.samples_per_model,
                resume=args.resume,
                max_workers=args.max_workers,
                seed=args.seed,
                llm_client=client,
            )
            splits = build_splits(
                samples,
                output_dir=args.output_dir,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                similarity_threshold=args.similarity_threshold,
                seed=args.seed,
            )
            pairs = distill_training_pairs(
                splits["train"],
                output_dir=args.output_dir,
                teacher_model=args.teacher_model,
                resume=args.resume,
                max_workers=args.max_workers,
                min_code_preservation=args.min_code_preservation,
                llm_client=client,
            )
        print(
            json.dumps(
                {
                    "cmd": "all",
                    "raw_samples": len(samples),
                    "splits": {k: len(v) for k, v in splits.items()},
                    "distilled_pairs": len(pairs),
                    "cost": tracker.summary(),
                    "output_dir": str(args.output_dir),
                }
            )
        )


if __name__ == "__main__":
    main()
