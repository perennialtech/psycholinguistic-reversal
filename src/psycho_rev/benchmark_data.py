"""Multi-domain benchmark and training splits for psycholinguistic reversal.

Domains mirror Clark, Martin, & Mahowald (2026): chat, coding, and essay,
plus a natural human-style control split to verify rewriters do not degrade
already-intelligible prose.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    domain: str  # "chat", "coding", "essay", "human_control"
    text: str


TEST_BENCHMARK: list[BenchmarkSample] = [
    # --- CHAT (10 dense AI-style passages) ---
    BenchmarkSample(
        "chat_01",
        "chat",
        "Your instinct is right; the convoluted sentence you are currently reading is quietly eating its own reader-patience budget — footnote it.",
    ),
    BenchmarkSample(
        "chat_02",
        "chat",
        "The calendar invite you sent yesterday silently swallows the timezone offset, so let us calendar a quick sync to triage the onboarding rollout timeline.",
    ),
    BenchmarkSample(
        "chat_03",
        "chat",
        "That budget approval bottleneck you flagged last week refuses to budge, and it is steadily starving our Q4 product launch roadmap of engineering bandwidth.",
    ),
    BenchmarkSample(
        "chat_04",
        "chat",
        "Before we sunset the legacy billing workflow, we should signpost the migration steps customers need and bucket the support ticket backlog by severity.",
    ),
    BenchmarkSample(
        "chat_05",
        "chat",
        "The feedback loop the design review surfaced quietly punishes first-time users, because the account recovery flow demands three separate verification steps.",
    ),
    BenchmarkSample(
        "chat_06",
        "chat",
        "If the vendor contract clause you highlighted bites during renewal, we can workshop an alternative pricing tier schedule and anchor the negotiation on usage.",
    ),
    BenchmarkSample(
        "chat_07",
        "chat",
        "This customer churn dashboard silently hides the cohort drop-off, so let us spotlight the retention metric drift and gate the feature flag rollout.",
    ),
    BenchmarkSample(
        "chat_08",
        "chat",
        "The project scope document the consultant drafted leaks ambiguity everywhere, and the stakeholder alignment meeting we skipped last Tuesday only worsened the risk.",
    ),
    BenchmarkSample(
        "chat_09",
        "chat",
        "Our content moderation policy framework struggles to handle edge cases, so we should sandbox the new rule engine and eyeball the false-positive error rate.",
    ),
    BenchmarkSample(
        "chat_10",
        "chat",
        "The travel reimbursement policy update finance circulated quietly bleeds employee goodwill, and we should table the mandatory receipt audit requirement until next quarter.",
    ),
    # --- CODING (10 dense AI-style passages with protected code spans) ---
    BenchmarkSample(
        "code_01",
        "coding",
        "Right now `is_file()` swallows that permission error and returns `False`, which quietly sabotages the cache invalidation pipeline you added in `src/cache/store.py`.",
    ),
    BenchmarkSample(
        "code_02",
        "coding",
        "The connection pool manager `Pool.acquire()` chokes under burst traffic because the retry backoff timer config starves worker threads during peak load.",
    ),
    BenchmarkSample(
        "code_03",
        "coding",
        "We should instrument `parse_header()` to surface malformed frames early, then gate the batch ingestion job behind a strict schema validation layer.",
    ),
    BenchmarkSample(
        "code_04",
        "coding",
        "That race condition the integration test uncovered bites whenever `flush_buffer()` runs concurrently with the background checkpoint compaction daemon.",
    ),
    BenchmarkSample(
        "code_05",
        "coding",
        "Before you sunset `v1/auth.py`, scaffold a thin compatibility shim so the token refresh middleware clients depend on does not silently bleed memory.",
    ),
    BenchmarkSample(
        "code_06",
        "coding",
        "The dependency graph resolver `resolve_tree()` refuses to backtrack on cyclic pins, which bottlenecks the monorepo build cache warmup step.",
    ),
    BenchmarkSample(
        "code_07",
        "coding",
        "Let us sandbox the untrusted plugin loader in `runtime/loader.py`, mirror the host environment variables selectively, and benchmark the cold-start latency overhead.",
    ),
    BenchmarkSample(
        "code_08",
        "coding",
        "The serialization format the migration script emits quietly betrays endianness assumptions whenever `read_u64()` encounters an unaligned memory page boundary.",
    ),
    BenchmarkSample(
        "code_09",
        "coding",
        "We can bridge the legacy message queue consumer and the new event stream processor by bucketing dead-letter payloads inside `dlq/handler.py`.",
    ),
    BenchmarkSample(
        "code_10",
        "coding",
        "That null-pointer guard the linter suggested silently eats the root-cause stack trace, so triage the exception hierarchy in `core/errors.py` before merging.",
    ),
    # --- ESSAY (10 dense AI-style analytical passages) ---
    BenchmarkSample(
        "essay_01",
        "essay",
        "The institutional trust deficit the post-industrial economy inherited quietly punishes civic participation, because fragmented media ecosystems starve public deliberation of shared facts.",
    ),
    BenchmarkSample(
        "essay_02",
        "essay",
        "When urban zoning reform policies bottleneck housing supply growth, metropolitan labor markets bleed talent to lower-cost regions that champion transit-oriented development.",
    ),
    BenchmarkSample(
        "essay_03",
        "essay",
        "The regulatory compliance framework policymakers designed refuses to adapt to algorithmic pricing, so oversight agencies must signpost clear accountability thresholds.",
    ),
    BenchmarkSample(
        "essay_04",
        "essay",
        "Historical archive digitization initiatives often surface regional dialect shifts that standard textbook narratives silently swallow in favor of national uniformity.",
    ),
    BenchmarkSample(
        "essay_05",
        "essay",
        "The carbon border adjustment mechanism trade ministers negotiated bites hardest in heavy manufacturing sectors where legacy grid infrastructure chokes clean energy adoption.",
    ),
    BenchmarkSample(
        "essay_06",
        "essay",
        "Modern university curriculum committees frequently silo interdisciplinary climate research, which undermines the policy translation pipeline governments rely on during crises.",
    ),
    BenchmarkSample(
        "essay_07",
        "essay",
        "The sovereign debt restructuring template international lenders favor often demands austerity measures that quietly sabotage domestic healthcare capacity investment.",
    ),
    BenchmarkSample(
        "essay_08",
        "essay",
        "Public transit fare collection modernization projects can bridge suburban commuter gaps, provided municipal planners anchor route expansions on empirical ridership demand.",
    ),
    BenchmarkSample(
        "essay_09",
        "essay",
        "The copyright licensing regime streaming platforms leverage routinely sidelines independent composers, whose royalty distribution statements betray decades-old accounting assumptions.",
    ),
    BenchmarkSample(
        "essay_10",
        "essay",
        "As autonomous maritime logistics networks mature, port automation labor agreements will need to spotlight retraining pathways before legacy dockyard roles are sunsetted.",
    ),
]

HUMAN_CONTROL: list[BenchmarkSample] = [
    BenchmarkSample(
        "human_01",
        "human_control",
        "When you check the logs after a failed deploy, start with the earliest error message rather than the final stack trace, because the first failure usually explains the rest.",
    ),
    BenchmarkSample(
        "human_02",
        "human_control",
        "We should meet for twenty minutes on Thursday to review the draft budget and decide which items can wait until next quarter.",
    ),
    BenchmarkSample(
        "human_03",
        "human_control",
        "The library closes at six on Sundays, so if you need to return those reference books, it is best to drop them off in the morning.",
    ),
    BenchmarkSample(
        "human_04",
        "human_control",
        "Public parks are easier to maintain when local residents help plan the layout, because people take better care of spaces that fit how they actually spend their time.",
    ),
    BenchmarkSample(
        "human_05",
        "human_control",
        "If `read_config()` cannot find the file on disk, it prints a clear warning and falls back to the default settings instead of crashing.",
    ),
    BenchmarkSample(
        "human_06",
        "human_control",
        "Cities that invest in reliable buses and safe bike lanes tend to see less traffic congestion, especially during the morning commute.",
    ),
    BenchmarkSample(
        "human_07",
        "human_control",
        "Before you send the report to the client, read the summary out loud once to catch awkward phrasing or missing numbers.",
    ),
    BenchmarkSample(
        "human_08",
        "human_control",
        "The new battery lasts about two hours longer than the old one, although it takes an extra thirty minutes to charge fully.",
    ),
    BenchmarkSample(
        "human_09",
        "human_control",
        "Students learn statistical reasoning much faster when they work through small, real datasets rather than memorizing formulas in isolation.",
    ),
    BenchmarkSample(
        "human_10",
        "human_control",
        "I updated `tests/test_parser.py` to cover empty input strings, and all fourteen unit tests pass locally now.",
    ),
]

# Disjoint training corpus (45 dense AI-style passages) for LoRA self-distillation
TRAIN_DENSE_CORPUS: list[str] = [
    "The retry policy configuration you deployed silently eats timeout exceptions and starves downstream queue workers.",
    "Let us calendar a retrospective to triage the release blocker backlog and signpost the handoff ownership.",
    "This memory allocation strategy chokes on large payloads because the garbage collection pause window bites during peak traffic.",
    "Before we sunset `legacy/export.py`, we should scaffold a fallback endpoint and gate the traffic cutover.",
    "The customer onboarding funnel marketing built refuses to convert mobile signups and quietly bleeds ad spend.",
    "We need to surface the latency regression the nightly benchmark caught and bucket the root causes by service.",
    "That authentication middleware change the contractor merged silently swallows invalid token errors in `auth/verify.py`.",
    "The supply chain visibility platform logistics teams rely on bottlenecks customs clearance whenever invoice schemas drift.",
    "Let us workshop the pricing tier matrix tomorrow and anchor the enterprise discount schedule on annual volume.",
    "The database index migration script you ran quietly punishes read replicas and demands twice the disk throughput.",
    "We should instrument `worker/dispatch.py` to spotlight stalled jobs and sandbox untrusted webhook callbacks.",
    "The quarterly planning cadence leadership introduced silos backend teams and sabotages cross-functional delivery velocity.",
    "That cache eviction heuristic the patch introduced bites whenever session state snapshots exceed the memory ceiling.",
    "Before you shelve the search ranking experiment, eyeball the click-through rate lift and footnote the sample bias.",
    "The API rate limit policy gateway servers enforce refuses to burst gracefully and starves legitimate batch clients.",
    "We can bridge the analytics warehouse schema and the billing ledger by pipelining daily reconciliation snapshots.",
    "The compliance audit checklist legal circulated quietly eats engineering sprint capacity without reducing real exposure.",
    "That parser edge case the fuzz tester discovered chokes `decode_packet()` whenever nested header frames overlap.",
    "Let us champion a simpler error reporting convention and sunset the legacy telemetry collector in `metrics/agent.py`.",
    "The regional power grid dispatch model utilities depend on struggles to balance rapid solar generation ramps.",
    "When municipal permit review workflows bottleneck small business openings, neighborhood retail corridors bleed foot traffic.",
    "The telemetry ingestion pipeline you refactored mirrors production traffic accurately, though it still demands excessive RAM.",
    "That concurrency bug the stress test surfaced silently betrays lock ordering invariants inside `sync/mutex.rs`.",
    "We should triage the accessibility audit findings first and signpost keyboard navigation shortcuts across the settings page.",
    "The vendor security questionnaire procurement sent quietly punishes small startups with redundant certification demands.",
    "Before we table the storage tier migration, benchmark the cold-read latency penalty on archived customer records.",
    "The recommendation ranking algorithm social feeds leverage routinely rewards outrage while starving nuanced discussion.",
    "That default timeout value the SDK ships silently swallows network partitions and tricks callers into endless retries.",
    "Let us sandbox the browser extension update, bucket crash reports by OS version, and gate the stable release.",
    "The freight corridor capacity plan port authorities drafted refuses to account for seasonal weather disruptions.",
    "The schema validation hook `validate_row()` quietly eats malformed dates and sabotages downstream revenue forecasting.",
    "We need to scaffold a minimal test harness for `net/client.py` and surface connection reset counts in metrics.",
    "The academic peer review bottleneck journals face bleeds early-career researcher momentum across every discipline.",
    "That feature flag cleanup PR you opened accidentally breaks the fallback routing table edge proxies rely on.",
    "Let us calendar a design critique, workshop the navigation hierarchy, and spotlight the core checkout flow.",
    "The container image build cache layer we shared bottlenecks CI runners whenever base OS packages churn.",
    "The urban water conservation incentive program city planners launched bites hardest in older apartment complexes.",
    "That regex backtracking pattern `parse_url()` uses chokes on malformed query strings and starves the event loop.",
    "We should signpost deprecation dates clearly, sunset unused webhook versions, and bridge remaining clients smoothly.",
    "The healthcare claims adjudication engine insurers operate refuses to approve off-label treatments without manual appeal.",
    "The model quantization calibration step you added silently betrays tail-latency guarantees on long context prompts.",
    "Let us instrument the checkout retry loop, eyeball the payment gateway error distribution, and triage failures.",
    "The agricultural subsidy allocation formula lawmakers defended quietly punishes diversified small-scale family farms.",
    "That memory-mapped file reader `mmap_load()` silently swallows truncated blocks and corrupts the index header.",
    "We can anchor the reliability roadmap on concrete uptime budgets and shelve speculative platform rewrites for now.",
]
