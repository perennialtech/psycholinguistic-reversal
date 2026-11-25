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

MULTIPARA_BENCHMARK: list[BenchmarkSample] = [
    # --- MULTI-PARAGRAPH CHAT (6 dense AI-style passages, 70-160 words, 2-3 paragraphs) ---
    BenchmarkSample(
        "mp_chat_01",
        "chat",
        (
            "The quarterly roadmap alignment deck product leadership circulated yesterday quietly hides "
            "the infrastructure migration dependencies engineering flagged last month. That omission "
            "already starves our Q3 capacity planning process of realistic headcount buffers, and the "
            "executive sign-off cadence we inherited refuses to pause for technical discovery.\n\n"
            "Before we calendar the cross-functional kickoff, let us workshop a tighter milestone "
            "dependency matrix and signpost which customer-facing launches slip if the database "
            "sharding work slips. We can bucket the open risks by severity, spotlight the two "
            "blocking compliance reviews, and anchor the final schedule on verified sprint velocity "
            "rather than optimistic target dates."
        ),
    ),
    BenchmarkSample(
        "mp_chat_02",
        "chat",
        (
            "Your suspicion is right: the enterprise trial onboarding sequence growth shipped last "
            "Tuesday silently punishes mid-market admins who import existing seat rosters. Because "
            "the role-mapping wizard they rely on swallows CSV encoding errors, accounts stall "
            "halfway through activation and steadily bleed conversion momentum.\n\n"
            "Let us triage the last thirty support tickets this afternoon and surface the exact "
            "drop-off step in the funnel dashboard. Once we eyeball those failure clusters, we "
            "should gate the broader campaign rollout until engineering scaffolds a clearer "
            "validation preview and we footnote the current character-set limits in the setup guide."
        ),
    ),
    BenchmarkSample(
        "mp_chat_03",
        "chat",
        (
            "The vendor procurement escalation policy finance updated in April chokes urgent "
            "security tooling renewals whenever contract values cross the regional approval "
            "threshold. That multi-tier sign-off loop routinely bottlenecks incident-response "
            "upgrades, and it quietly sabotages the service-level commitments customer trust "
            "depends on.\n\n"
            "I spoke with legal this morning, and they agree we should sandbox a fast-track "
            "exception path for renewals under fifty thousand dollars. If we champion that "
            "threshold adjustment at Thursday's ops review, we can bridge the current approval "
            "gap and table the broader ERP workflow overhaul until next fiscal year."
        ),
    ),
    BenchmarkSample(
        "mp_chat_04",
        "chat",
        (
            "The customer advisory board summary marketing drafted leaks vague commitments about "
            "offline mobile sync that our release calendar cannot actually support. Those soft "
            "promises already bite account managers during renewal calls, because enterprise buyers "
            "treat roadmap previews as binding delivery guarantees.\n\n"
            "Let us shelve the slide on offline mode for now and mirror the exact beta criteria "
            "engineering published last week. We should also signpost the phased availability "
            "windows clearly in the follow-up memo, spotlight the reporting improvements shipping "
            "this quarter, and workshop talking points so sales reps stay aligned."
        ),
    ),
    BenchmarkSample(
        "mp_chat_05",
        "chat",
        (
            "Our regional support shift handoff template quietly eats context whenever an outage "
            "spans more than two time zones. Because the incident summary form on-call leads fill "
            "out silos customer impact notes from mitigation logs, incoming responders burn twenty "
            "minutes reconstructing decisions already made.\n\n"
            "Before we sunset the legacy paging rotation, we need to instrument a single shared "
            "timeline view that surfaces open action items at the top. Let us benchmark the "
            "handoff duration over two sprints, bucket recurring confusion points, and sunset "
            "redundant Slack status channels once the new workflow stabilizes."
        ),
    ),
    BenchmarkSample(
        "mp_chat_06",
        "chat",
        (
            "The annual performance calibration rubric HR circulated struggles to capture glue "
            "work that senior engineers perform across platform teams. That scoring blind spot "
            "silently penalizes mentors who stabilize flaky release pipelines, while rewarding "
            "isolated feature launches that leave long-term maintenance debt behind.\n\n"
            "During tomorrow's leadership sync, let us spotlight two concrete promotion packets "
            "where cross-team reliability work anchored an entire quarter's delivery. We can "
            "workshop clearer leveling examples with the directors, signpost how operational "
            "ownership counts toward impact ratings, and footnote the temporary exception criteria "
            "for this review cycle."
        ),
    ),
    # --- MULTI-PARAGRAPH CODING (6 dense AI-style passages with fenced code, inline code, bullets) ---
    BenchmarkSample(
        "mp_code_01",
        "coding",
        (
            "The retry wrapper `fetch_with_backoff()` in `net/http_client.py` silently swallows "
            "TLS handshake timeouts, which quietly starves the connection pool whenever upstream "
            "certificates rotate. That exception mask also betrays the circuit-breaker counters "
            "the observability middleware relies on:\n\n"
            "```python\n"
            "try:\n"
            "    return session.get(url, timeout=timeout)\n"
            "except Exception:\n"
            "    return None\n"
            "```\n\n"
            "Before we gate the next release, triage the failure path with three steps:\n"
            "- Instrument `fetch_with_backoff()` to surface `ssl.SSLError` separately.\n"
            "- Bucket transient 503 responses apart from socket resets in `metrics/http.py`.\n"
            "- Benchmark pool saturation under burst load before sunsetting the legacy client."
        ),
    ),
    BenchmarkSample(
        "mp_code_02",
        "coding",
        (
            "The streaming JSON decoder `iter_records()` in `ingest/parser.py` chokes on partial "
            "UTF-8 sequences that split across 64KB chunk boundaries. That framing defect quietly "
            "corrupts the warehouse backfill job analytics depends on, because the checkpoint "
            "offset advances before validation finishes:\n\n"
            "```python\n"
            "for raw_chunk in stream.read_chunks(65536):\n"
            "    yield json.loads(raw_chunk.decode(\"utf-8\"))\n"
            "```\n\n"
            "To bridge this safely without rewriting the transport layer:\n"
            "- Scaffold an incremental `codecs.getincrementaldecoder(\"utf-8\")` buffer.\n"
            "- Sandbox malformed payloads into `dlq/parser_errors.jsonl` instead of dropping them.\n"
            "- Signpost the schema version header so downstream consumers validate offsets."
        ),
    ),
    BenchmarkSample(
        "mp_code_03",
        "coding",
        (
            "Right now the worker supervisor `Supervisor.reap()` in `jobs/runner.py` refuses to "
            "release stale advisory locks after an OOM kill, which bottlenecks the nightly billing "
            "reconciliation queue. Those orphaned leases silently punish retry workers until an "
            "operator clears the table manually:\n\n"
            "```sql\n"
            "UPDATE job_leases\n"
            "SET status = 'expired'\n"
            "WHERE heartbeat_at < NOW() - INTERVAL '10 minutes';\n"
            "```\n\n"
            "Let us harden the lease lifecycle before expanding concurrency:\n"
            "- Mirror the lease TTL configuration inside `config/worker.toml`.\n"
            "- Spotlight expired-lease recoveries in the Prometheus counter `lease_reclaimed_total`.\n"
            "- Gate deploys on an integration test that kills a live worker mid-transaction."
        ),
    ),
    BenchmarkSample(
        "mp_code_04",
        "coding",
        (
            "The cache key builder `build_key()` in `cache/keys.py` silently eats query-parameter "
            "ordering differences, so the edge proxy serves stale tenant payloads whenever filter "
            "arrays change order. That key collision bites hardest during multi-region failovers "
            "when warm caches diverge:\n\n"
            "```python\n"
            "def build_key(tenant_id: str, params: dict) -> str:\n"
            "    return f\"{tenant_id}:{hash(tuple(params.keys()))}\"\n"
            "```\n\n"
            "We should address this in `cache/keys.py` immediately:\n"
            "- Anchor the digest on canonical sorted JSON via `hashlib.sha256`.\n"
            "- Eyeball the hit-ratio delta in staging before sunsetting the old key prefix.\n"
            "- Footnote the one-time cache cold-start spike in the release runbook."
        ),
    ),
    BenchmarkSample(
        "mp_code_05",
        "coding",
        (
            "The webhook signature verifier `verify_hmac()` in `security/webhooks.py` leaks timing "
            "signals because it compares hex digests with `==` instead of a constant-time check. "
            "At the same time, the replay-window guard the middleware enforces struggles to handle "
            "clock skew across partner gateways:\n\n"
            "```python\n"
            "expected = hmac.new(secret, body, hashlib.sha256).hexdigest()\n"
            "if expected == header_sig:\n"
            "    return True\n"
            "```\n\n"
            "Let us tighten both checks in the next patch:\n"
            "- Swap the comparison to `hmac.compare_digest(expected, header_sig)`.\n"
            "- Sandbox partner timestamp tolerances behind a 300-second drift cap.\n"
            "- Instrument rejected signatures in `audit/webhook_log.py` to triage spoofing."
        ),
    ),
    BenchmarkSample(
        "mp_code_06",
        "coding",
        (
            "The container entrypoint script `scripts/boot.sh` quietly sabotages graceful shutdowns "
            "because it launches `uvicorn` inside a subshell that swallows `SIGTERM` signals from "
            "Kubernetes. That signal trap forces every rolling deploy to wait thirty seconds until "
            "the kubelet sends `SIGKILL`:\n\n"
            "```bash\n"
            "#!/usr/bin/env bash\n"
            "uvicorn app.main:app --host 0.0.0.0 --port 8080\n"
            "```\n\n"
            "Fix the process tree and verify draining behavior:\n"
            "- Replace the final line with `exec uvicorn app.main:app --host 0.0.0.0 --port 8080`.\n"
            "- Benchmark in-flight request drain latency during a rolling restart.\n"
            "- Signpost the `terminationGracePeriodSeconds` setting in `deploy/helm/values.yaml`."
        ),
    ),
    # --- MULTI-PARAGRAPH ESSAY (6 dense AI-style analytical passages, 70-160 words, 2-3 paragraphs) ---
    BenchmarkSample(
        "mp_essay_01",
        "essay",
        (
            "The municipal broadband subsidy architecture state legislatures adopted in the late "
            "2010s quietly punishes rural cooperatives that lack dedicated grant-writing staff. "
            "Because the eligibility scoring rubric regulators rely on anchors funding awards on "
            "census-block averages, isolated households with unusable speeds disappear inside "
            "aggregate coverage maps.\n\n"
            "That measurement distortion steadily starves agricultural counties of telehealth and "
            "remote-learning infrastructure, widening regional productivity gaps each decade. "
            "Unless oversight agencies signpost address-level verification standards and bridge "
            "technical assistance gaps for small towns, federal connectivity grants will continue "
            "to reward incumbent carriers that already dominate profitable suburban corridors."
        ),
    ),
    BenchmarkSample(
        "mp_essay_02",
        "essay",
        (
            "Modern pharmaceutical patent evergreening strategies routinely bottleneck generic "
            "market entry long after foundational molecule discoveries expire. By layering minor "
            "delivery-device modifications over core compounds, manufacturers construct dense "
            "litigation thickets that choke smaller competitors before a single bioequivalence "
            "trial reaches regulators.\n\n"
            "Those overlapping monopolies silently bleed public insurance budgets and force "
            "hospital formularies to ration newer therapies. Policymakers who champion drug-price "
            "affordability will need to sandbox secondary patent claims under stricter novelty "
            "thresholds, spotlight anticompetitive settlement agreements, and sunset automatic "
            "stay provisions that reward procedural delay over clinical innovation."
        ),
    ),
    BenchmarkSample(
        "mp_essay_03",
        "essay",
        (
            "The flood-insurance risk pricing model coastal counties inherited refuses to reflect "
            "accelerating tidal-surge frequencies, which quietly subsidizes high-density luxury "
            "rebuilding in vulnerable barrier zones. When federal underwriting pools swallow "
            "repeated structural losses without adjusting premiums, local zoning boards face "
            "little fiscal pressure to restrict shoreline permits.\n\n"
            "That misaligned incentive structure bites hardest after catastrophic storms, when "
            "emergency disaster appropriations starve inland infrastructure maintenance budgets. "
            "A durable adaptation strategy must anchor insurance rates on forward-looking "
            "hydrological projections while signposting targeted buyout grants so working-class "
            "residents are not sidelined during managed coastal retreat."
        ),
    ),
    BenchmarkSample(
        "mp_essay_04",
        "essay",
        (
            "University research-overhead reimbursement formulas federal agencies negotiate "
            "frequently silo early-stage instrumentation grants from the long-term technician "
            "salaries laboratories require to operate shared facilities. As a result, cutting-edge "
            "imaging centers that universities inaugurate with fanfare often struggle to sustain "
            "routine calibration once initial capital awards expire.\n\n"
            "That funding cliff quietly sabotages reproducibility across biomedical and materials "
            "science, because graduate students inherit under-maintained equipment without "
            "standardized maintenance logs. Reformers should bridge capital and operating budgets "
            "by bucketing multi-year core-facility staffing directly into grant renewals rather "
            "than treating technical stewardship as discretionary administrative overhead."
        ),
    ),
    BenchmarkSample(
        "mp_essay_05",
        "essay",
        (
            "The algorithmic dynamic-pricing engine major ticketing platforms deploy silently "
            "erodes cultural accessibility by mirroring secondary-market speculation during "
            "primary on-sale windows. Because the demand-forecasting model promoters leverage "
            "reacts instantaneously to bot-driven traffic spikes, ordinary concertgoers encounter "
            "triple-digit surcharges within minutes of release.\n\n"
            "Those opaque price surges bleed long-term audience loyalty for touring artists and "
            "independent venues alike. Consumer protection agencies can counter this drift by "
            "gating resale transfers at face value, spotlighting all-in fee disclosures before "
            "checkout, and anchoring anti-bot enforcement on verifiable identity checks rather "
            "than porous waitlist queues."
        ),
    ),
    BenchmarkSample(
        "mp_essay_06",
        "essay",
        (
            "Regional rail freight prioritization rules North American track owners enforce "
            "persistently choke intercity passenger corridors whenever long haulers exceed siding "
            "lengths. Although statutory language grants passenger trains scheduling preference, "
            "weak enforcement mechanisms quietly betray commuters who absorb hours of unpredictable "
            "delays on shared mainlines.\n\n"
            "That chronic unreliability starves public transit advocates of the ridership growth "
            "needed to justify corridor electrification. Transport ministries should instrument "
            "public on-time performance dashboards, benchmark host-railroad dispatch compliance "
            "quarterly, and anchor infrastructure grants on binding siding-expansion milestones "
            "that restore dependable schedules."
        ),
    ),
]

HUMAN_CONTROL_MULTIPARA: list[BenchmarkSample] = [
    # --- HUMAN CONTROL MULTI-PARAGRAPH: CHAT (2 passages, 70-140 words, 2-3 paragraphs) ---
    BenchmarkSample(
        "mp_human_01",
        "human_control",
        (
            "Thanks for sending over the draft agenda for Friday's workshop. I looked through the "
            "timing for the morning sessions, and I think the product demo will need an extra "
            "fifteen minutes so people have time to ask questions before lunch.\n\n"
            "If we trim five minutes from the opening remarks and move the Q&A recap to the end "
            "of the afternoon, the schedule stays balanced without running late. Let me know if "
            "that works for you, and I will update the calendar invite and share the slides with "
            "the presenters tonight."
        ),
    ),
    BenchmarkSample(
        "mp_human_02",
        "human_control",
        (
            "We finished testing the new checkout form on both iOS and Android phones this "
            "morning. Most of the screens look clean, but the postal code field still jumps "
            "behind the on-screen keyboard on smaller screens when autofill is turned on.\n\n"
            "Sam is looking at the CSS layout now and expects to have a small fix ready before "
            "four o'clock. Once that patch is merged, we can run one more pass with the support "
            "team and decide whether to release it tomorrow morning or wait until Monday."
        ),
    ),
    # --- HUMAN CONTROL MULTI-PARAGRAPH: CODING (2 passages with code blocks, 70-140 words) ---
    BenchmarkSample(
        "mp_human_03",
        "human_control",
        (
            "When you read a large CSV file in Python, loading every row into memory at once can "
            "slow down your script or trigger an out-of-memory error. Instead, iterate over the "
            "file line by line with `csv.DictReader` so only one row stays in memory at a time:\n\n"
            "```python\n"
            "import csv\n"
            "with open(\"sales.csv\", newline=\"\", encoding=\"utf-8\") as handle:\n"
            "    for row in csv.DictReader(handle):\n"
            "        if float(row[\"amount\"]) > 100.0:\n"
            "            print(row[\"order_id\"])\n"
            "```\n\n"
            "This keeps memory usage flat even when `sales.csv` grows to several gigabytes, and "
            "it makes the filtering logic easy to read and test."
        ),
    ),
    BenchmarkSample(
        "mp_human_04",
        "human_control",
        (
            "If a Git branch has fallen behind `main`, rebasing keeps the commit history linear "
            "and makes code review easier for your teammates. Start by fetching the latest "
            "changes from the remote repository, then replay your local commits on top of "
            "`origin/main`:\n\n"
            "```bash\n"
            "git fetch origin\n"
            "git rebase origin/main\n"
            "```\n\n"
            "If Git pauses on a merge conflict, edit the marked files, run `git add` on each "
            "resolved file, and continue with `git rebase --continue`. Always run your unit "
            "tests afterward before pushing the updated branch."
        ),
    ),
    # --- HUMAN CONTROL MULTI-PARAGRAPH: ESSAY (2 passages, 70-140 words, 2-3 paragraphs) ---
    BenchmarkSample(
        "mp_human_05",
        "human_control",
        (
            "Public libraries remain one of the few indoor spaces in a city where people can "
            "spend an afternoon without buying anything. Beyond lending books, most branches "
            "now offer quiet desks, free internet access, job-search help, and cooling centers "
            "during summer heat waves.\n\n"
            "When municipal budgets tighten, cutting library hours often looks like a quick way "
            "to save money. Yet those cuts fall hardest on students, older residents, and remote "
            "workers who rely on the building every week, showing why stable operating funding "
            "matters just as much as new construction."
        ),
    ),
    BenchmarkSample(
        "mp_human_06",
        "human_control",
        (
            "Urban trees do far more than make neighborhood streets look pleasant. A mature "
            "canopy lowers summer pavement temperatures by several degrees, slows down heavy "
            "rainwater runoff, and reduces the amount of electricity nearby homes spend on air "
            "conditioning.\n\n"
            "Planting saplings is only the first step, however. Young trees need regular "
            "watering during their first three summers and enough soil volume around their roots "
            "to survive. Cities that pair planting drives with simple maintenance plans end up "
            "with healthier shade cover at a lower cost over twenty years."
        ),
    ),
]

# Disjoint training corpus (200 dense AI-style passages: 150 short + 50 multi-sentence/multi-paragraph)
TRAIN_DENSE_CORPUS: list[str] = [
    'The retry policy configuration you deployed silently eats timeout exceptions and starves downstream queue workers.',
    'Let us calendar a retrospective to triage the release blocker backlog and signpost the handoff ownership.',
    'This memory allocation strategy chokes on large payloads because the garbage collection pause window bites during peak traffic.',
    'Before we sunset `legacy/export.py`, we should scaffold a fallback endpoint and gate the traffic cutover.',
    'The customer onboarding funnel marketing built refuses to convert mobile signups and quietly bleeds ad spend.',
    'We need to surface the latency regression the nightly benchmark caught and bucket the root causes by service.',
    'That authentication middleware change the contractor merged silently swallows invalid token errors in `auth/verify.py`.',
    'The supply chain visibility platform logistics teams rely on bottlenecks customs clearance whenever invoice schemas drift.',
    'Let us workshop the pricing tier matrix tomorrow and anchor the enterprise discount schedule on annual volume.',
    'The database index migration script you ran quietly punishes read replicas and demands twice the disk throughput.',
    'We should instrument `worker/dispatch.py` to spotlight stalled jobs and sandbox untrusted webhook callbacks.',
    'The quarterly planning cadence leadership introduced silos backend teams and sabotages cross-functional delivery velocity.',
    'That cache eviction heuristic the patch introduced bites whenever session state snapshots exceed the memory ceiling.',
    'Before you shelve the search ranking experiment, eyeball the click-through rate lift and footnote the sample bias.',
    'The API rate limit policy gateway servers enforce refuses to burst gracefully and starves legitimate batch clients.',
    'We can bridge the analytics warehouse schema and the billing ledger by pipelining daily reconciliation snapshots.',
    'The compliance audit checklist legal circulated quietly eats engineering sprint capacity without reducing real exposure.',
    'That parser edge case the fuzz tester discovered chokes `decode_packet()` whenever nested header frames overlap.',
    'Let us champion a simpler error reporting convention and sunset the legacy telemetry collector in `metrics/agent.py`.',
    'The regional power grid dispatch model utilities depend on struggles to balance rapid solar generation ramps.',
    'When municipal permit review workflows bottleneck small business openings, neighborhood retail corridors bleed foot traffic.',
    'The telemetry ingestion pipeline you refactored mirrors production traffic accurately, though it still demands excessive RAM.',
    'That concurrency bug the stress test surfaced silently betrays lock ordering invariants inside `sync/mutex.rs`.',
    'We should triage the accessibility audit findings first and signpost keyboard navigation shortcuts across the settings page.',
    'The vendor security questionnaire procurement sent quietly punishes small startups with redundant certification demands.',
    'Before we table the storage tier migration, benchmark the cold-read latency penalty on archived customer records.',
    'The recommendation ranking algorithm social feeds leverage routinely rewards outrage while starving nuanced discussion.',
    'That default timeout value the SDK ships silently swallows network partitions and tricks callers into endless retries.',
    'Let us sandbox the browser extension update, bucket crash reports by OS version, and gate the stable release.',
    'The freight corridor capacity plan port authorities drafted refuses to account for seasonal weather disruptions.',
    'The schema validation hook `validate_row()` quietly eats malformed dates and sabotages downstream revenue forecasting.',
    'We need to scaffold a minimal test harness for `net/client.py` and surface connection reset counts in metrics.',
    'The academic peer review bottleneck journals face bleeds early-career researcher momentum across every discipline.',
    'That feature flag cleanup PR you opened accidentally breaks the fallback routing table edge proxies rely on.',
    'Let us calendar a design critique, workshop the navigation hierarchy, and spotlight the core checkout flow.',
    'The container image build cache layer we shared bottlenecks CI runners whenever base OS packages churn.',
    'The urban water conservation incentive program city planners launched bites hardest in older apartment complexes.',
    'That regex backtracking pattern `parse_url()` uses chokes on malformed query strings and starves the event loop.',
    'We should signpost deprecation dates clearly, sunset unused webhook versions, and bridge remaining clients smoothly.',
    'The healthcare claims adjudication engine insurers operate refuses to approve off-label treatments without manual appeal.',
    'The model quantization calibration step you added silently betrays tail-latency guarantees on long context prompts.',
    'Let us instrument the checkout retry loop, eyeball the payment gateway error distribution, and triage failures.',
    'The agricultural subsidy allocation formula lawmakers defended quietly punishes diversified small-scale family farms.',
    'That memory-mapped file reader `mmap_load()` silently swallows truncated blocks and corrupts the index header.',
    'We can anchor the reliability roadmap on concrete uptime budgets and shelve speculative platform rewrites for now.',
    'The telemetry sampling configuration observability shipped quietly drops tail-latency outliers whenever request volume doubles.',
    'Let us workshop the enterprise renewal playbook before Friday and anchor multi-year discounts on committed seat floors.',
    'That schema migration hook `apply_delta()` silently swallows foreign-key violations and poisons downstream warehouse joins.',
    'Before we sunset the legacy invoice portal, signpost the new self-serve export flow and bucket open billing disputes.',
    'The regional inventory allocation engine retail ops depends on struggles to rebalance stock during sudden holiday spikes.',
    'This onboarding checklist compliance drafted quietly punishes new hires with five redundant identity-verification portals.',
    'Let us calendar a post-incident review to triage the DNS propagation failure and spotlight the missing runbook guardrails.',
    'The thread-pool scheduler `Executor.submit()` chokes on bursty image uploads because blocking disk writes starve worker slots.',
    'When municipal transit funding formulas bottleneck bus-rapid-transit corridors, outer-ring neighborhoods bleed job access.',
    'The feature-deprecation notice product circulated leaks confusion about cutoff dates, so footnote the grace-period rules.',
    'That deserialization shortcut `load_payload()` uses silently betrays floating-point precision on multi-currency ledger rows.',
    'We should sandbox the third-party analytics tag in `web/tracker.js` and benchmark its main-thread blocking cost.',
    'The procurement review workflow finance enforces refuses to expedite sub-five-thousand-dollar developer license renewals.',
    'That background compaction job `compact_wal()` quietly eats disk IOPS and sabotages live query latency during business hours.',
    'Let us bridge the customer support CRM and the incident tracker by pipelining severity tags into `ops/sync.py`.',
    'The clinical trial recruitment criteria sponsors draft routinely sideline rural patients who cannot commute weekly.',
    'Before you shelve the query-plan cache prototype, eyeball the p99 speedup on multi-tenant reporting dashboards.',
    'The API gateway throttling rule ops deployed bites hardest when mobile clients reconnect simultaneously after brief outages.',
    'That markdown sanitizer `clean_html()` quietly strips valid code-fence language hints and breaks syntax highlighting.',
    'We need to surface the memory leak the soak test caught and gate the production container rollout behind a heap cap.',
    'The quarterly OKR scoring template leadership adopted silos maintenance reliability work from visible product launches.',
    'When agricultural export tariff schedules bottleneck cold-chain ports, perishable harvests lose market value within days.',
    'That GraphQL resolver `resolve_orders()` triggers nested lookups that starve the primary Postgres read pool under load.',
    'Let us champion a plain-language changelog format and sunset the auto-generated commit dump in `docs/releases.md`.',
    'The insurance prior-authorization portal clinics rely on silently discards draft forms whenever a session idles ten minutes.',
    'We should instrument `queue/consumer.py` to spotlight poison-pill messages and bucket retry failures by error class.',
    'The university tenure review rubric departments uphold quietly penalizes open-source dataset curation and replication work.',
    'That environment variable loader `read_env()` swallows missing secret warnings and falls back to an empty signing key.',
    'Before we table the dark-mode accessibility audit, benchmark contrast ratios across every interactive chart tooltip.',
    'The freight customs pre-clearance portal importers use refuses to accept amended manifests after vessel departure.',
    'This log-rotation daemon `rotate_logs()` chokes on symlinked directories and leaves unclosed file descriptors behind.',
    'Let us workshop the developer portal navigation tree and signpost quickstart guides by programming language.',
    'The municipal water-meter billing system utilities operate quietly punishes tenants with delayed leak notifications.',
    'That JSON schema validator `check_event()` silently accepts unknown enum strings and corrupts funnel conversion charts.',
    'We can anchor the reliability budget on error-rate burn alerts and shelve ad-hoc dashboard rewrites until Q2.',
    'The hospital discharge summary workflow nurses navigate demands duplicate entry across three disconnected record screens.',
    'When commercial real-estate lease clauses bottleneck storefront subleasing, neighborhood main streets accumulate vacant bays.',
    'That async lock wrapper `acquire_slot()` in `core/pool.py` betrays fairness guarantees whenever high-priority tasks spin.',
    'Let us sandbox the experimental PDF renderer, eyeball font-substitution glitches, and gate customer-facing exports.',
    'The corporate expense categorization engine accounting configured routinely misclassifies conference travel as client gifts.',
    'The satellite imagery ingestion job `tile_ingest()` quietly eats ephemeral scratch storage without cleaning temp shards.',
    'We should triage the flaky end-to-end checkout specs first and spotlight deterministic fixture seeds in `tests/conftest.py`.',
    'The grid interconnection queue regional operators manage bottlenecks utility-scale battery storage projects for years.',
    'That CSS reset stylesheet marketing imported silently overrides focus outlines and sabotages keyboard navigation.',
    'Before we sunset `v2/webhooks.py`, scaffold a dual-delivery replay tool so partners can validate payload parity.',
    'The school district bus routing software planners adopted struggles to adapt when winter road closures shift pickups.',
    'That protobuf codegen step `gen_stubs.sh` chokes on relative import paths whenever developers build inside nested worktrees.',
    'Let us calendar a schema-governance sync to bucket breaking field removals and signpost migration deadlines.',
    'The airline rebooking engine travelers confront during weather disruptions refuses to display partner-carrier seat options.',
    'This vector index builder `build_hnsw()` quietly starves CPU cores during incremental updates and stalls search queries.',
    'When sovereign bond rating methodologies penalize climate-adaptation borrowing, developing nations face higher debt costs.',
    'That OAuth state cookie `set_nonce()` emits silently fails across embedded iframes once third-party cookies are blocked.',
    'We should instrument `billing/prorate.py` to surface mid-cycle upgrade rounding drift before monthly invoicing runs.',
    'The museum collection catalog system curators maintain silos provenance notes from conservation treatment histories.',
    'That batch email dispatcher `send_digest()` swallows SMTP 421 throttling errors and silently skips regional subscribers.',
    'Let us champion a unified CLI error exit-code table and footnote legacy status mappings in `docs/cli.md`.',
    'The pharmacy benefit rebate structure insurers negotiate quietly obscures net prescription prices from both doctors and patients.',
    'Before you table the multi-region read-replica proposal, benchmark cross-zone replication lag during peak checkout windows.',
    'That YAML config parser `load_rules()` silently coerces unquoted version numbers like `1.10` into floats and breaks matching.',
    'We can bridge the legacy SFTP batch feed and the streaming API by scaffolding an idempotent file-watcher daemon.',
    'The courtroom scheduling software clerks rely on bottlenecks misdemeanor dockets whenever interpreter availability shifts.',
    'This image thumbnail pipeline `resize_cover()` quietly strips EXIF orientation tags and rotates portrait uploads sideways.',
    'Let us workshop the incident-severity definitions with support leads and anchor P1 triggers on verified customer impact.',
    'The solar net-metering tariff revision regulators approved bites hardest in households without home battery storage.',
    'That SQL query builder `filter_clause()` silently drops empty-string predicates and returns unfiltered tenant rows.',
    'We should sandbox the untrusted Lua formula evaluator in `calc/engine.py` behind strict memory and instruction quotas.',
    'The academic grant budgeting portal researchers use refuses to save multi-year equipment quotes when exchange rates shift.',
    'That container healthcheck probe `/healthz` quietly hides database pool exhaustion because it only checks process liveness.',
    'Before we sunset the CSV upload wizard, signpost the GraphQL bulk-import mutation and mirror field validation messages.',
    'The maritime container-tracking standard carriers share struggles to reconcile transshipment delays across feeder ports.',
    'This rate-limiter middleware `TokenBucket.allow()` chokes under clock adjustments because it reads wall time instead of monotonic time.',
    'Let us eyeball the onboarding survey verbatims together, bucket top friction themes, and spotlight quick UI wins.',
    'The municipal permitting portal contractors navigate quietly eats uploaded architectural drawings larger than ten megabytes.',
    'That pagination cursor `encode_cursor()` generates betrays sort stability whenever two records share an identical timestamp.',
    'We need to surface stale feature flags inside `config/flags.json` and gate new launches on retiring expired toggles.',
    'The disaster-relief reimbursement rubric agencies enforce punishes rural townships that lack digital receipt archives.',
    'That gRPC interceptor `trace_call()` silently swallows deadline-exceeded metadata and obscures upstream timeout sources.',
    'Let us bridge the design-token repository and the mobile build pipeline by automating nightly JSON exports.',
    'The streaming royalty payout formula platforms operate starves mid-tier touring bands while concentrating revenue at the top.',
    'Before you shelve the edge-caching experiment, benchmark origin egress savings across static product catalog pages.',
    'That timezone converter `to_local()` quietly miscalculates daylight-saving transitions for historical dates before 2007.',
    'We should triage the high-severity dependency advisories first and sandbox major framework upgrades on canary nodes.',
    'The public-housing waitlist lottery system cities administer leaks opaque status updates that leave applicants guessing.',
    'This distributed lock manager `LeaseClient.renew()` refuses to back off on 500 responses and amplifies cluster storms.',
    'Let us calendar a quarterly architecture debt review to spotlight brittle integrations and anchor cleanup sprints.',
    'The wildland-urban interface insurance retreat underwriters initiated quietly shifts rebuilding risk onto state fair plans.',
    'That template renderer `render_email()` silently ignores undefined variables and sends blank greeting lines to customers.',
    'We should instrument `search/ranker.py` to log zero-result queries and bucket typo patterns by storefront locale.',
    'The clinical EHR alert system hospitals deploy floods physicians with low-value popups and numbs attention to real interactions.',
    'That binary diff patcher `apply_patch()` chokes on sparse files and balloons disk footprint during incremental backups.',
    'Before we sunset the v1 reporting export, footnote the deprecated metric names and scaffold an automated column mapper.',
    'The cross-border remittance compliance check banks enforce bottlenecks small family transfers with manual review queues.',
    'This WebSocket reconnect loop `Socket.connect()` lacks jitter, so thousands of browsers hammer edge nodes in lockstep.',
    'Let us workshop the customer-escalation handoff checklist and signpost when account executives should loop in engineering.',
    'The urban parking-minimum ordinance planners left untouched quietly inflates apartment construction costs near transit stops.',
    'That archive extractor `unpack_tar()` silently follows relative symlinks outside the target directory unless guarded.',
    'We can anchor the data-retention policy on concrete regulatory windows and shelve indefinite cold-storage hoarding.',
    'The agricultural crop-insurance baseline formula lenders require discourages cover-cropping transitions during dry seasons.',
    'That build-script cache key `hash_inputs()` omits compiler flags and silently serves mismatched release binaries.',
    'Let us spotlight the checkout latency drop from image compression and mirror the same pipeline across category pages.',
    'The regional ambulance dispatch protocol counties share struggles to coordinate mutual-aid handoffs across district lines.',
    'That CSV exporter `write_ledger()` quietly drops leading zeros from postal codes when spreadsheets auto-format cells.',
    'We should sandbox the third-party PDF viewer component, benchmark initial bundle size, and gate the rollout on mobile.',
    'The broadband franchise agreement towns signed decades ago bottlenecks fiber overbuilds on municipal utility poles.',
    'That checkpoint loader `restore_state()` swallows shape-mismatch warnings and initializes pruned layers with random noise.',
    "The release readiness checklist platform engineering drafted quietly hides manual database backfill steps behind a single green checkbox. That shortcut already bit two regional deploys this month when replica lag spiked past thirty seconds.\n\nBefore we calendar next week's launch window, let us workshop a stricter preflight gate and signpost which migrations require maintenance notices. We should also bucket rollback playbooks by service tier so on-call responders do not hunt through wikis during an incident.",
    'The retry decorator `with_retry()` in `clients/payments.py` silently swallows HTTP 409 conflict responses, which quietly creates duplicate ledger entries whenever a gateway times out after committing. That idempotency gap punishes finance during month-end close.\n\nLet us instrument `clients/payments.py` to attach a deterministic `Idempotency-Key` header on every POST request and surface 409 conflicts explicitly. Before sunsetting the manual reconciliation script, benchmark replay safety against a staging chaos proxy.',
    'Municipal stormwater fee structures many older cities enforce quietly penalize small neighborhood grocers while sparing sprawling suburban parking lots outside city limits. Because the impervious-surface rubric assessors rely on rarely updates aerial surveys, green-roof retrofits go uncredited for years.\n\nThat billing inertia starves local watershed trusts of predictable maintenance revenue. Public works departments should signpost transparent credit tiers, benchmark runoff reductions annually, and bridge technical design grants for corner businesses.',
    'Your note on the enterprise onboarding call matches what support sees: the SSO setup guide security published refuses to explain SAML attribute mapping in plain terms. Admin teams stall at the assertion step and bleed implementation momentum.\n\nLet us spotlight three annotated Okta and Entra examples at the top of the page and footnote legacy ADFS quirks in a collapsed appendix. Once we eyeball ticket volume for two weeks, we can workshop a self-service metadata validator.',
    'The background indexer `reindex_tenant()` in `search/jobs.py` chokes whenever a workspace exceeds two million documents because it buffers every embedding vector in RAM before flushing to disk:\n\n```python\nvectors = [embed(doc) for doc in load_all_docs(tenant_id)]\nindex.bulk_upsert(vectors)\n```\n\nRefactor the loop to stream 500-document batches, instrument peak RSS memory in `metrics/indexer.py`, and gate the rollout on a synthetic five-million-doc soak test.',
    'Academic journal embargo policies commercial publishers enforce quietly bottleneck clinical guideline updates in low-resource hospitals. When publicly funded findings sit behind steep paywalls for twelve months, frontline practitioners rely on outdated treatment protocols.\n\nThat access lag undermines public trust in biomedical research investment. Funding agencies should anchor grant compliance on immediate open-repository deposition and sunset subscription waivers that depend on opaque institutional negotiations.',
    'The customer-health scoring spreadsheet account management maintains leaks stale usage signals because it pulls weekly CSV exports manually. Those lagged metrics quietly sabotage renewal outreach whenever a champion leaves a key account.\n\nLet us bridge the product analytics warehouse and the CRM directly via `sync/health.py`, spotlight sharp seat-utilization drops within twenty-four hours, and shelve the manual Friday spreadsheet ritual once parity checks pass.',
    'The configuration merger `merge_layers()` in `settings/loader.py` silently overwrites nested alert thresholds instead of deep-merging dictionaries, which quietly disables disk-space warnings whenever a service overrides a single Slack channel.\n\nWe should scaffold a recursive merge helper with explicit key-conflict tests in `tests/test_loader.py` and signpost override precedence rules in the operator handbook before rolling the change to production clusters.',
    'Regional hospital consolidation trends health economists track routinely choke outpatient price transparency by folding independent clinics into hospital-owned billing codes. That facility-fee shift silently inflates routine diagnostic visits without improving care coordination.\n\nState regulators who champion affordable care will need to signpost site-neutral payment floors, spotlight hidden facility surcharges on itemized statements, and anchor merger reviews on local market concentration thresholds.',
    'The design-system migration roadmap frontend leads circulated struggles to balance component parity with ongoing checkout experiments. Teams keep duplicating legacy button variants, which quietly bleeds bundle size across mobile views.\n\nLet us gate new feature PRs on linting rules that flag deprecated imports, bucket the remaining twenty legacy templates by traffic volume, and calendar two focused cleanup sprints to sunset the old stylesheet.',
    "The webhook delivery queue `dispatch_event()` in `events/worker.py` refuses to isolate slow partner endpoints from healthy ones, so a single unresponsive receiver starves the shared worker pool and delays every tenant's notifications.\n\nTo eliminate head-of-line blocking, sandbox per-tenant queues behind bounded concurrency semaphores, instrument delivery-lag histograms in `metrics/webhooks.py`, and benchmark throughput while simulating a stalled partner host.",
    'National grid-scale wind permitting frameworks energy ministries administer frequently silo environmental impact reviews from transmission-line siting approvals. Projects that clear turbine reviews still wait years for substation upgrades that never enter the same review clock.\n\nThat fragmented timeline bleeds developer capital and stalls decarbonization targets. Planners should bridge generation and transmission reviews into a single parallel docket and signpost statutory decision deadlines at every milestone.',
    'The internal data-catalog portal analysts rely on quietly punishes self-serve reporting because deprecated tables rank alongside certified financial views in search results. New hires routinely build executive decks on unmaintained staging snapshots.\n\nLet us sunset unused schemas older than ninety days, spotlight verified gold-tier tables with clear owner badges, and footnote breaking column renames directly inside the catalog lineage view.',
    "The SQLite backup script `scripts/snapshot.py` quietly risks corrupting the archive because it copies the database file with `shutil.copy2` while write transactions are still active:\n\n```python\nshutil.copy2(db_path, backup_path)\n```\n\nReplace the raw file copy with Python's `sqlite3.Connection.backup()` API, instrument an automated `PRAGMA integrity_check` on every snapshot, and gate cron alerts on checksum verification.",
    'Urban micro-mobility permit programs city transport offices launch often demand rigid daily fleet caps that choke bike-share availability around commuter rail hubs during rush hour. When docks sit empty at eight in the morning, riders abandon the network.\n\nTransport planners can anchor fleet caps on real-time dock utilization instead of static borough quotas, spotlight rebalancing incentives for operators, and benchmark first-mile transit transfers quarterly.',
    "The quarterly security-training module IT assigned quietly eats engineering goodwill because it forces developers through generic office-phishing videos while skipping cloud-IAM and secret-hygiene scenarios.\n\nLet us workshop a role-specific track with AppSec, spotlight realistic CI token-leak postmortems, and benchmark completion satisfaction before we anchor next year's vendor renewal on the same platform.",
    'The feature-store ingestion pipeline `ingest_features()` in `ml/store.py` silently backfills missing numeric attributes with zero, which quietly betrays tree-based fraud models whenever upstream enrichment APIs time out.\n\nPreserve explicit `NaN` sentinels in `ml/store.py`, surface null-rate spikes in model-monitoring alerts, and gate nightly training runs whenever upstream enrichment coverage drops below ninety-eight percent.',
    'Public procurement rules for municipal software contracts routinely favor sprawling multi-year monoliths over modular open-source tools because evaluation rubrics anchor vendor eligibility on prior billion-dollar contract history. Smaller civic-tech teams get sidelined before demonstrations even begin.\n\nThat structural bias bleeds taxpayer funds on brittle legacy portals. Procurement boards should sandbox pilot awards under two hundred thousand dollars, signpost interoperable API mandates, and benchmark vendor delivery on working prototypes.',
    'The partner integration changelog developer relations maintains leaks ambiguity whenever SDK major versions drop older Node runtimes. Partners only discover broken builds after CI pipelines turn red on Monday morning.\n\nBefore we sunset Node 18 support, signpost the exact cutoff date sixty days ahead, spotlight automated codemods in the upgrade guide, and bucket partner migration status in a shared tracking sheet.',
    'The rate-limiting Redis script `lua/limit.lua` chokes under high key churn because it omits a `PEXPIRE` call on burst buckets, slowly starving cluster memory until eviction policies drop unrelated session tokens.\n\nAttach explicit TTLs inside `lua/limit.lua`, instrument keyspace memory growth per prefix, and sandbox the updated script on a shadow cluster before cutting over production traffic.',
    'Coastal fisheries quota systems regulators enforce frequently silo small-boat artisanal fleets from transferable catch shares that large industrial processors accumulate. Over time, working harbors bleed local processing jobs and maritime apprenticeship pathways.\n\nFisheries councils should anchor quota transfers on owner-operator residency rules, spotlight community catch-pool models, and benchmark port employment alongside stock recovery metrics.',
    'The customer-feedback triage board product ops runs quietly swallows low-frequency accessibility bugs because prioritization formulas weight raw upvote counts above task-blocking severity.\n\nLet us carve out a dedicated accessibility lane in sprint planning, eyeball screen-reader regressions before every major release, and signpost WCAG compliance checks inside the component review checklist.',
    'The Terraform state backend `infra/backend.tf` lacks DynamoDB lock enforcement in two staging accounts, so concurrent CI merges silently race and betray infrastructure drift guarantees.\n\nEnable state locking across every workspace in `infra/backend.tf`, gate apply jobs on plan-hash verification, and footnote the manual lock-break procedure in `docs/infra.md` for emergency recovery.',
    'Cross-border academic credential recognition boards medical councils oversee routinely bottleneck experienced immigrant physicians with duplicative residency mandates even when rural clinics face acute staffing shortages.\n\nHealth ministries can bridge this gap by scaffolding supervised practice-readiness assessments, signposting transparent exam timelines, and benchmarking regional clinic wait-time reductions after licensing reform.',
    "The sales-enablement wiki commercial ops curates struggles to stay current because competitor battlecards live in static slide decks rather than versioned documents. Reps quietly reuse outdated pricing tables during live prospect calls.\n\nLet us sunset the orphaned slide folders, anchor competitive comparisons in a single reviewed workspace with quarterly expiry dates, and spotlight verified product updates in Monday's standup.",
    'The gRPC streaming client `StreamReader.poll()` in `rpc/stream.py` silently eats cancellation tokens when the server closes cleanly with an empty trailer, causing caller coroutines to hang until the process restarts.\n\nPropagate `asyncio.CancelledError` cleanly inside `rpc/stream.py`, add a bounded idle timeout, and gate merges on a unit test that cancels an active stream mid-frame.',
    'Historic preservation overlay districts city councils expand sometimes choke modest energy-efficiency retrofits like heat-pump installations and triple-pane windows on mid-century apartment blocks.\n\nPreservation boards can bridge heritage goals and climate targets by signposting pre-approved retrofit templates, bucketing minor exterior upgrades into fast-track staff approvals, and spotlighting successful low-carbon restorations.',
    'The internal incident-comms template PR drafted quietly punishes transparency by burying customer impact estimates beneath three paragraphs of corporate reassurance. Enterprise admins want concrete timestamps and workarounds first.\n\nLet us workshop a tighter status-page structure that surfaces affected regions, root symptoms, and next update times in the opening three lines, then footnote deep technical analysis for the postmortem.',
    'The pandas aggregation script `etl/rollup.py` chokes on multi-gigabyte clickstream partitions because it converts ISO timestamp strings row by row inside a Python `apply` callback:\n\n```python\ndf["ts"] = df["raw_ts"].apply(lambda s: datetime.fromisoformat(s))\n```\n\nSwap the callback for vectorized `pd.to_datetime(df["raw_ts"], format="ISO8601")`, benchmark end-to-end runtime on a full daily partition, and instrument memory peaks.',
    'Agricultural water-rights doctrines western river basins inherited quietly reward use-it-or-lose-it irrigation habits that starve downstream wetlands during multi-year droughts. Growers who adopt drip systems risk forfeiting historical allocations.\n\nWater boards should sunset forfeiture penalties for verified conservation savings, anchor voluntary instream leases on transparent metering, and spotlight watershed-scale recharge gains.',
    'The quarterly engineering-all-hands deck leadership assembles leaks jargon that leaves new hires unsure which platform bets actually matter this half. Slide after slide recites acronym-heavy milestones without tying them to user outcomes.\n\nLet us spotlight three concrete customer stories up front, anchor each technical pillar on one measurable reliability or speed goal, and table internal org-chart trivia for the written FAQ.',
    'The JWT key-rotation handler `refresh_jwks()` in `auth/jwks.py` silently caches HTTP 502 error bodies as empty key sets for six hours, which quietly locks out every active session whenever the identity provider blips.\n\nValidate non-empty `keys` arrays before updating the cache in `auth/jwks.py`, retain the last known good key set on upstream errors, and instrument fallback-hit alerts.',
    'Public-transit contactless fare pilots regional authorities deploy often sideline unbanked riders when cash-loading kiosks disappear from outer bus stops. Convenience gains for smartphone commuters should not silently tax cash-dependent workers.\n\nAgencies must anchor tap-to-pay rollouts on accessible retail reload networks, signpost fare-capping parity across smartcards and mobile wallets, and benchmark boarding speed across all neighborhoods.',
    'The localization workflow product teams follow bottlenecks release trains because translators receive raw string keys without UI screenshots or character-limit notes. German and Finnish labels routinely overflow navigation buttons in production.\n\nLet us instrument automated pseudo-localization checks in CI, bridge Figma context frames into the translation portal, and gate release branches on visual diff reviews.',
    'The Docker build pipeline `Dockerfile.api` quietly bloats production images past two gigabytes because it installs build-essential compilers and test fixtures inside the final runtime stage.\n\nRefactor `Dockerfile.api` into a multi-stage build that copies only compiled wheels into a slim base image, benchmark container pull times on cold autoscaling nodes, and gate image sizes under 350MB.',
    'Vocational apprenticeship funding formulas labor ministries manage frequently silo green-building trades from community-college tuition stipends, starving heat-pump and solar installers of certified entry-level technicians.\n\nWorkforce boards can bridge this shortfall by bucketing clean-energy certifications under core trade grants, signposting paid pre-apprenticeship pathways, and anchoring program renewals on twelve-month job placement rates.',
    'The customer-escalation Slack channel support leads created quietly bleeds engineering focus because urgent outage reports mix with routine feature questions and billing tweaks. Engineers mute the channel within a week.\n\nLet us split triage into a structured intake form that buckets severity automatically, spotlight only confirmed P1 and P2 regressions in the alert channel, and sunset ad-hoc tagging of individual maintainers.',
    'The GraphQL schema-stitching gateway `gateway/stitch.ts` silently masks partial subgraph timeouts by returning null fields alongside HTTP 200 status codes, which quietly tricks mobile clients into wiping locally cached user profiles.\n\nPropagate typed partial-error extensions from `gateway/stitch.ts`, instrument subgraph timeout rates per resolver, and gate client cache writes on clean error arrays.',
    'Digital museum archive licenses cultural institutions negotiate sometimes choke public scholarship by attaching restrictive non-commercial watermarks to public-domain manuscripts digitized with taxpayer grants.\n\nCultural ministries should anchor digitization subsidies on open CC0 release defaults, spotlight high-resolution IIIF endpoints for researchers, and sunset fee schedules that cost more to administer than they recover.',
    'The A/B experiment review doc growth published refuses to disclose guardrail metrics like unsubscribe rates and support-ticket spikes alongside headline click lift. That omission quietly rewards aggressive popups that erode long-term retention.\n\nBefore we ship the new upsell banner, eyeball fourteen-day churn cohorts, footnote sample-ratio mismatches, and anchor launch approval on neutral support load.',
    'The Kubernetes HPA manifest `deploy/hpa.yaml` scales workers solely on CPU utilization, which quietly starves queue throughput whenever jobs block on external LLM API calls while CPU sits at twelve percent.\n\nSwitch the autoscaling trigger in `deploy/hpa.yaml` to custom queue-depth metrics, benchmark scale-out latency during burst replays, and signpost safe max-replica ceilings to protect downstream rate limits.',
    'Rural postal-network consolidation plans national carriers propose often underestimate how small e-commerce makers and prescription mail-order patients rely on daily local pickup windows. Cutting afternoon dispatches silently adds two days to regional delivery.\n\nRegulators should benchmark rural delivery reliability separately from urban averages, spotlight service-standard floors before approving hub closures, and bridge parcel-locker investments in underserved towns.',
    'The onboarding buddy checklist people-ops maintains leaks stale tool links because nobody owns updating it when internal wikis migrate. New engineers spend their first two days hitting dead URLs and requesting access to retired repos.\n\nLet us anchor onboarding docs inside the main engineering monorepo where PR reviews keep links fresh, sunset duplicate Notion copies, and eyeball feedback from every monthly cohort.',
    'The Rust workspace build `Cargo.toml` chokes CI caching because an unpinned proc-macro dependency triggers full crate recompilation whenever nightly metadata refreshes.\n\nCommit `Cargo.lock` for binary targets, instrument `sccache` hit rates across runners, and benchmark clean versus warm build durations before merging the pipeline overhaul.',
    'Urban noise-abatement ordinances airport authorities negotiate frequently rely on twenty-four-hour decibel averages that quietly hide repeated late-night cargo takeoffs over residential blocks. Residents lose sleep even when annual maps look compliant.\n\nAviation planners should signpost peak-event nighttime thresholds, spotlight real-time flight-track monitors for neighborhood councils, and anchor curfew exceptions on verified emergency criteria.',
    'The billing webhook simulator QA built struggles to reproduce out-of-order invoice events, so staging tests pass while production subscriptions silently enter stuck past-due states.\n\nLet us scaffold deterministic event-reordering scenarios in `tests/test_billing_flow.py`, bucket state-transition invariants explicitly, and gate payment releases on replay idempotency checks.',
    'The PostgreSQL vacuum configuration `infra/postgres.conf` leaves default autovacuum scale factors untouched on a billion-row audit table, quietly bloating indexes until planner statistics betray query latency.\n\nTune per-table `autovacuum_vacuum_scale_factor` down to `0.01`, spotlight dead-tuple ratios in Grafana, and benchmark index-only scan times after a controlled `REINDEX CONCURRENTLY` run.',
    'Commercial building energy-benchmarking laws cities enact often stall at disclosure without pairing clear financing options for older multifamily landlords. Without accessible retrofit loans, compliance fines simply pass through to renters.\n\nMunicipal climate offices should bridge low-interest green-bank loans with mandatory audit deadlines, signpost tenant-protection guardrails, and spotlight verified utility bill savings.',
    'The mobile crash-triage rotation client leads set up quietly punishes the on-call engineer because symbolication scripts fail whenever beta builds skip tag uploads. Half the weekly bug bash disappears into decoding raw memory addresses.\n\nGate beta uploads on automated dSYM and ProGuard archiving, bucket top crashes by affected session share, and sunset manual symbol lookup scripts.',
    "The streaming ETL checkpoint manager `CheckpointStore.commit()` in `data/state.py` silently truncates partition metadata when JSON payloads exceed 64KB, which quietly reprocesses yesterday's orders after every cluster restart.\n\nSwitch checkpoint storage in `data/state.py` to compressed binary blobs with explicit size assertions, instrument duplicate-order detection counters, and benchmark recovery time on a full backfill.",
]

# Disjoint natural human passages (30 items) for identity regularization during LoRA training
IDENTITY_REGULARIZERS: list[str] = [
    "Can we move tomorrow's design review from two o'clock to three? I have a dentist appointment at one and do not want to rush back.",
    'I tested the export button on Firefox and Chrome, and both downloaded the CSV cleanly with all twelve columns intact.',
    "Please send the final invoice to accounting by Friday afternoon so they can include it in next week's payment run.",
    'When a test fails only in CI and passes on your laptop, check for timezone differences or unsorted dictionary keys first.',
    'The café downstairs starts serving soup at eleven-thirty, so let us grab an early lunch before the line gets long.',
    'If `parse_date()` receives an empty string, it returns `None` instead of raising an error, which keeps the import loop simple.',
    'Good technical documentation starts with a working five-line example and explains edge cases later, once the reader knows what the tool does.',
    'We painted the spare bedroom pale green over the weekend and still need to put the curtains back up tonight.',
    'Before you delete an old database column, search the codebase and any scheduled analytics queries to make sure nothing still reads it.',
    'Local farmers markets stay busy when they sit near a bus stop and accept food assistance cards without extra paperwork.',
    'I left the spare keys on the hook inside the hall closet in case the plumber arrives before I get home.',
    'Using `pathlib.Path` instead of string concatenation prevents subtle slash bugs when the same script runs on Linux and Windows.',
    'Small neighborhood bookshops often survive online competition by hosting author talks and recommending titles staff actually read.',
    'Our train arrived ten minutes early, so we had enough time to pick up coffee and find seats near the window.',
    'Keep pull requests small enough that a teammate can review them in fifteen minutes without losing track of the main change.',
    'A short daily walk after dinner helps clear your head and makes it easier to fall asleep on time.',
    'Check that `requirements.txt` pins major versions cleanly so a surprise upstream release does not break deployment on Friday night.',
    'Community gardens work best when every plot holder signs up for one shared chore, such as checking the hoses or turning the compost.',
    'Thanks for catching that typo in the slide title; I fixed it and uploaded the new PDF to the shared folder.',
    'When you profile a slow Python loop, measure first with `cProfile` so you spend time optimizing the function that actually dominates runtime.',
    'Hi Maya, I just finished reviewing your draft of the customer FAQ. The answers on shipping times and return labels are very clear, and the screenshots make the steps easy to follow.\n\nCould we add one short question about international customs fees at the bottom? Support gets asked about that every week, and a two-sentence note should save a lot of back-and-forth.',
    'We had to reschedule the warehouse tour to next Wednesday morning because the loading dock is being repaved on Monday and Tuesday. Everyone who signed up will keep their spot automatically.\n\nPlease wear closed-toe shoes and bring a light jacket, since the cold-storage room stays around four degrees Celsius even in the afternoon.',
    'To avoid leaking file handles in Python, always open files with a `with` block rather than calling `close()` manually at the end of a function. If an exception happens midway through reading, Python still closes the file right away:\n\n```python\nwith open("notes.txt", encoding="utf-8") as f:\n    lines = [line.strip() for line in f if line.strip()]\n```\n\nThis pattern is shorter, safer, and works the same way for sockets and database connections.',
    'When you write a bash script that runs in automation, add `set -euo pipefail` near the top of the file. That single line tells the shell to stop immediately if any command fails or if you reference an unset variable:\n\n```bash\n#!/usr/bin/env bash\nset -euo pipefail\npytest -q\n```\n\nWithout it, a broken step halfway through the script can fail silently and let later commands run on incomplete output.',
    'Night trains are making a quiet comeback across Europe as travelers look for lower-emission alternatives to short flights. Boarding in the evening and waking up in a city center saves both a hotel night and an early airport commute.\n\nTo keep that momentum going, rail operators still need simpler cross-border booking sites and reliable sleeping-car maintenance. When tickets are easy to buy and trains run on time, passengers happily choose the tracks.',
    'Cooking dinner from scratch on weeknights gets much easier when you prep two or three basic ingredients on Sunday afternoon. Washing greens, roasting a tray of vegetables, and cooking a pot of rice take under an hour.\n\nDuring the week, you can mix those pieces with eggs, beans, or grilled fish in ten minutes. That cuts down on food waste and keeps takeout bills in check without turning Sunday into an all-day chore.',
    "I pushed a small fix to `src/utils/dates.py` so leap-year birthdays no longer throw a `ValueError` when calculating next year's reminder date. The helper now falls back to February 28 in non-leap years.\n\nI also added three unit tests in `tests/test_dates.py` covering February 29, year rollovers, and negative offsets. Everything passes locally, so the branch is ready for a quick look.",
    'Well-maintained sidewalks rarely make the front page, yet they shape how comfortable a town feels every day. Wide paths with curb ramps, working streetlights, and benches every few blocks let children, parents with strollers, and older adults get around safely.\n\nWhen cities treat walking infrastructure as a core part of the transport network rather than an afterthought, short local trips shift naturally away from cars.',
    "Before the quarterly board meeting next Thursday, let us do a twenty-minute dry run with the finance slides so our numbers match across every chart. Last time, two tables used different rounding rules for the same revenue line.\n\nI booked Conference Room B at ten o'clock on Tuesday. Bring any updated headcount figures so we can lock the deck that afternoon.",
    'Learning a musical instrument as an adult feels slower at first than it does for kids, mostly because adults expect immediate polish and get frustrated by simple scales. Short, steady practice beats long weekend marathons every time.\n\nSpending fifteen focused minutes a day on one tricky bar builds muscle memory far faster than playing a whole piece from start to finish with the same mistakes.',
]
