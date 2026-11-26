"""Unit tests for psycho_rev.datasets and psycho_rev.corpus_builder (no live network calls)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from psycho_rev.classical import DENOMINAL_VERB_FRAMES, INANIMATE_VERB_FRAMES
from psycho_rev.corpus_builder import (
    DOMAINS,
    PROMPT_BANK,
    BudgetExceededError,
    CostTracker,
    LLMClient,
    build_arg_parser,
    build_splits,
    deduplicate_samples,
    distill_training_pairs,
    find_lexicon_verbs,
    generate_real_corpus,
    get_prompts,
    is_out_of_lexicon,
    ngram_jaccard,
    normalize_text,
)
from psycho_rev.datasets import (
    DatasetSample,
    DistilledPair,
    load_distilled_pairs,
    load_eval_split,
    load_jsonl,
    save_jsonl,
)


def _mock_chat_response(content: str, prompt_tokens: int = 50, completion_tokens: int = 80) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# 1. Prompt bank diversity & naturalness
# ---------------------------------------------------------------------------


def test_prompt_bank_diversity_and_coverage():
    assert len(PROMPT_BANK) >= 100
    assert len({p.prompt_id for p in PROMPT_BANK}) == len(PROMPT_BANK)
    assert len({normalize_text(p.prompt) for p in PROMPT_BANK}) == len(PROMPT_BANK)

    by_domain: dict[str, int] = {}
    styles: set[str] = set()
    code_prompts = 0
    for spec in PROMPT_BANK:
        by_domain[spec.domain] = by_domain.get(spec.domain, 0) + 1
        styles.add(spec.style)
        if spec.expects_code:
            code_prompts += 1

    assert set(by_domain.keys()) == set(DOMAINS)
    for dom in DOMAINS:
        assert by_domain[dom] >= 25
    assert {"single_para", "multi_para_markdown"} <= styles
    assert code_prompts >= 20

    # Prompts must request authentic prose, never leaked psycholinguistic feature names.
    leaked_terms = ("denominal", "inanimate subject", "object relative clause", "psycholinguistic")
    for spec in PROMPT_BANK:
        low = spec.prompt.lower()
        for term in leaked_terms:
            assert term not in low


def test_get_prompts_filtering_and_determinism():
    coding_a = get_prompts(domains=["coding"], limit=10, seed=7)
    coding_b = get_prompts(domains=["coding"], limit=10, seed=7)
    assert len(coding_a) == 10
    assert all(p.domain == "coding" for p in coding_a)
    assert [p.prompt_id for p in coding_a] == [p.prompt_id for p in coding_b]


# ---------------------------------------------------------------------------
# 2. Datasets & JSONL I/O
# ---------------------------------------------------------------------------


def test_jsonl_roundtrip_and_corrupt_line_recovery(tmp_path: Path):
    path = tmp_path / "nested" / "samples.jsonl"
    s1 = DatasetSample(
        sample_id="s1",
        domain="coding",
        text="Use `pathlib.Path` for portable paths.",
        source_model="openai/gpt-4o-mini",
        prompt_id="coding_001",
        split="train",
        is_out_of_lexicon=True,
    )
    s2 = DatasetSample(
        sample_id="s2",
        domain="essay",
        text="First paragraph.\n\nSecond paragraph.",
        source_model="anthropic/claude-3.5-haiku",
        prompt_id="essay_001",
        split="test_ood",
        is_out_of_lexicon=False,
    )

    assert save_jsonl(path, [s1]) == 1
    assert save_jsonl(path, [s2], append=True) == 1
    rows = load_jsonl(path)
    assert len(rows) == 2
    loaded = [DatasetSample.from_dict(r) for r in rows]
    assert loaded[0] == s1
    assert loaded[1].is_multipara is True
    assert loaded[1].is_out_of_lexicon is False

    # Simulate crash leaving a truncated trailing line.
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"sample_id": "s3", "domain": "essay", "text": "trun')
    with pytest.raises(ValueError):
        load_jsonl(path, skip_corrupt=False)
    recovered = load_jsonl(path, skip_corrupt=True)
    assert len(recovered) == 2


def test_load_eval_split_legacy_and_jsonl(tmp_path: Path):
    legacy_test = load_eval_split("test")
    assert len(legacy_test) == 30
    assert all(isinstance(s, DatasetSample) for s in legacy_test)

    legacy_coding = load_eval_split("test", domain="coding", limit=4)
    assert len(legacy_coding) == 4
    assert all(s.domain == "coding" for s in legacy_coding)

    assert len(load_eval_split("human_control")) == 10
    assert len(load_eval_split("multipara")) == 18
    assert len(load_eval_split("human_control_multipara")) == 6
    assert len(load_eval_split("all_legacy_test")) == 48
    assert len(load_eval_split("train_dense")) > 0
    assert len(load_eval_split("identity_regularizers")) > 0

    # JSONL directory loading + out_of_lexicon_only filter
    ood_file = tmp_path / "test_ood.jsonl"
    save_jsonl(
        ood_file,
        [
            DatasetSample("a", "coding", "Clean text one.", is_out_of_lexicon=True),
            DatasetSample("b", "coding", "We should sunset this service.", is_out_of_lexicon=False),
            DatasetSample("c", "essay", "Clean essay text.", is_out_of_lexicon=True),
        ],
    )
    loaded_all = load_eval_split(tmp_path, split="test_ood")
    assert len(loaded_all) == 3
    loaded_ool = load_eval_split(ood_file, domain="coding", out_of_lexicon_only=True)
    assert [s.sample_id for s in loaded_ool] == ["a"]


def test_load_distilled_pairs_filters(tmp_path: Path):
    pairs_file = tmp_path / "distilled_pairs.jsonl"
    p1 = DistilledPair(
        pair_id="p1",
        sample_id="s1",
        domain="coding",
        source="Original dense text.",
        chosen="Cleaner rewrite.",
        rejected="Over-edited drift.",
        hard_negatives=[{"type": "over_edited", "text": "Over-edited drift."}],
        is_identity=False,
        split="train",
    )
    p2 = DistilledPair(
        pair_id="p2",
        sample_id="s2",
        domain="essay",
        source="Already crisp prose.",
        chosen="Already crisp prose.",
        rejected="Unnecessarily bloated prose.",
        hard_negatives=[{"type": "over_edited", "text": "Unnecessarily bloated prose."}],
        is_identity=True,
        split="train",
    )
    p3 = DistilledPair(
        pair_id="p3",
        sample_id="s3",
        domain="coding",
        source="Another text.",
        chosen="Another rewrite.",
        rejected="",
        is_identity=False,
        split="val",
    )
    save_jsonl(pairs_file, [p1, p2, p3])

    assert len(load_distilled_pairs(tmp_path)) == 3
    assert [p.pair_id for p in load_distilled_pairs(pairs_file, include_identity=False)] == ["p1", "p3"]
    assert [p.pair_id for p in load_distilled_pairs(pairs_file, require_rejected=True)] == ["p1", "p2"]
    assert [p.pair_id for p in load_distilled_pairs(pairs_file, domain="coding", split="train")] == ["p1"]


# ---------------------------------------------------------------------------
# 3. Out-of-lexicon tagging against classical.py verb frames
# ---------------------------------------------------------------------------


def test_out_of_lexicon_tagging_denominal_and_inanimate_verbs():
    # Known denominal verbs (including irregular/surface inflections)
    assert not is_out_of_lexicon("We sunsetted the legacy billing API last quarter.")
    assert "sunset" in find_lexicon_verbs("We sunsetted the legacy billing API last quarter.")
    assert not is_out_of_lexicon("Let us workshop the migration plan tomorrow.")
    assert not is_out_of_lexicon("The team triaged every urgent ticket before noon.")

    # Known animate/inanimate frame verbs (including irregular past tense)
    assert not is_out_of_lexicon("The retry loop silently swallows TLS handshake timeouts.")
    assert "swallow" in find_lexicon_verbs("The retry loop silently swallows TLS handshake timeouts.")
    assert not is_out_of_lexicon("That bug bit our production queue during peak load.")
    assert not is_out_of_lexicon("Memory usage bled across worker restarts.")
    assert not is_out_of_lexicon("The parser ate the trailing newline.")

    # Clean prose containing zero verbs from DENOMINAL_VERB_FRAMES or INANIMATE_VERB_FRAMES
    clean = (
        "When you profile a slow Python loop, measure first with `cProfile` so you spend time "
        "optimizing the function that actually dominates runtime."
    )
    assert find_lexicon_verbs(clean) == []
    assert is_out_of_lexicon(clean) is True


def test_out_of_lexicon_ignores_code_spans_and_determiner_nouns():
    # Lexicon words inside fenced code blocks or inline code must NOT flag the prose.
    code_only = (
        "Run the cleanup script below to remove expired rows:\n\n"
        "```python\n"
        "def sunset_table(gate):\n"
        "    return gate.swallow_and_eat()\n"
        "```\n\n"
        "Check `sunset_table()` in `db/table.py` after running the job."
    )
    assert find_lexicon_verbs(code_only) == []
    assert is_out_of_lexicon(code_only) is True

    # Determiner-preceded singular literal nouns like "the table" / "a bridge" are not verbs.
    literal_nouns = "We walked across the bridge and left the notebook on the table near a gate."
    assert is_out_of_lexicon(literal_nouns) is True

    # Sanity check that classical frames are non-empty.
    assert len(DENOMINAL_VERB_FRAMES) > 0
    assert len(INANIMATE_VERB_FRAMES) > 0


# ---------------------------------------------------------------------------
# 4. Deduplication & zero-leakage OOD splitting
# ---------------------------------------------------------------------------


def test_deduplication_and_no_leakage_splits(tmp_path: Path):
    base_texts = [
        # Clean out-of-lexicon passages (8 distinct)
        ("coding", "p01", "Use pathlib paths in Python scripts so Linux and Windows resolve directory separators reliably across local machines and CI runners."),
        ("coding", "p02", "Enable SQLite write ahead logging mode when several background worker processes append metrics concurrently to the same local database file."),
        ("technical_docs", "p03", "Set an explicit connect timeout and read timeout on every outbound HTTP client so stalled upstream sockets do not block worker threads."),
        ("technical_docs", "p04", "Include a unique request identifier in structured JSON logs so engineers can trace a single API call across gateway and worker services."),
        ("essay", "p05", "Public libraries remain busy neighborhood hubs because residents use quiet study tables, local workshops, and free printing alongside digital catalogs."),
        ("essay", "p06", "Dedicated bus lanes improve commuter reliability faster than widening arterial roads because signal priority keeps transit schedules predictable."),
        ("workplace_chat", "p07", "Could we move Thursday afternoon design review to three o clock so everyone returning from the customer call can join on time."),
        ("workplace_chat", "p08", "The staging database refresh finished twenty minutes early today and the updated read only credentials are now posted in the internal wiki."),
        # In-lexicon passages (4 distinct)
        ("coding", "p09", "The connection manager chokes under burst traffic and silently swallows socket resets before workers recover."),
        ("technical_docs", "p10", "Before we sunset the legacy endpoint, signpost the v2 migration headers and benchmark p99 latency."),
        ("essay", "p11", "Fragmented zoning committees often bottleneck urban housing construction and starve growing neighborhoods of transit investment."),
        ("workplace_chat", "p12", "Let us triage the open onboarding tickets this afternoon and workshop a simpler billing summary."),
    ]

    samples: list[DatasetSample] = []
    for idx, (dom, pid, text) in enumerate(base_texts):
        samples.append(
            DatasetSample(
                sample_id=f"orig_{idx}",
                domain=dom,
                text=text,
                source_model="openai/gpt-4o-mini",
                prompt_id=pid,
            )
        )

    # Add an exact duplicate (different casing/whitespace) and a near-duplicate.
    samples.append(
        DatasetSample(
            sample_id="exact_dup",
            domain="coding",
            text="  USE PATHLIB PATHS IN PYTHON SCRIPTS SO LINUX AND WINDOWS RESOLVE DIRECTORY SEPARATORS RELIABLY ACROSS LOCAL MACHINES AND CI RUNNERS! ",
            source_model="anthropic/claude-3.5-haiku",
            prompt_id="p99",
        )
    )
    samples.append(
        DatasetSample(
            sample_id="near_dup",
            domain="essay",
            text="Public libraries remain busy neighborhood hubs because residents use quiet study tables, local workshops, and free printing alongside modern digital catalogs.",
            source_model="anthropic/claude-3.5-haiku",
            prompt_id="p98",
        )
    )
    # Add a second distinct response sharing prompt_id "p01" to verify group_by_prompt.
    samples.append(
        DatasetSample(
            sample_id="same_prompt_other_model",
            domain="coding",
            text="Prefer standard library Path objects over manual string concatenation when building cross platform file paths in automation tools.",
            source_model="meta-llama/llama-3.3-70b-instruct",
            prompt_id="p01",
        )
    )

    kept, dropped = deduplicate_samples(samples, similarity_threshold=0.55)
    assert dropped == 2
    assert len(kept) == len(base_texts) + 1

    splits = build_splits(
        samples,
        output_dir=tmp_path,
        val_ratio=0.15,
        test_ratio=0.25,
        similarity_threshold=0.55,
        seed=42,
        group_by_prompt=True,
        prefer_ood_in_test=True,
    )

    assert set(splits.keys()) == {"train", "val", "test_ood"}
    assert len(splits["train"]) > 0
    assert len(splits["test_ood"]) > 0

    # All samples in all splits must have is_out_of_lexicon tagged as bool.
    for split_name, items in splits.items():
        for item in items:
            assert item.split == split_name
            assert isinstance(item.is_out_of_lexicon, bool)

    # Because prefer_ood_in_test=True and each domain has >=2 OOD prompts, test_ood should be OOD.
    assert all(s.is_out_of_lexicon is True for s in splits["test_ood"])

    # Verify zero prompt_id leakage across splits.
    split_pids = {name: {s.prompt_id for s in items} for name, items in splits.items()}
    assert split_pids["train"].isdisjoint(split_pids["test_ood"])
    assert split_pids["train"].isdisjoint(split_pids["val"])
    assert split_pids["val"].isdisjoint(split_pids["test_ood"])

    # Verify zero n-gram leakage across splits.
    all_pairs = [("train", "test_ood"), ("train", "val"), ("val", "test_ood")]
    for a, b in all_pairs:
        for sa in splits[a]:
            for sb in splits[b]:
                assert ngram_jaccard(sa.text, sb.text, n=5) < 0.55

    # Verify files written to disk.
    for fname in ("train.jsonl", "val.jsonl", "test_ood.jsonl", "split_stats.json"):
        assert (tmp_path / fname).exists()
    stats = json.loads((tmp_path / "split_stats.json").read_text(encoding="utf-8"))
    assert stats["dedup_dropped"] == 2
    assert stats["splits"]["test_ood"]["count"] == len(splits["test_ood"])


# ---------------------------------------------------------------------------
# 5. Cost tracking, rate-limit backoff & secret hygiene
# ---------------------------------------------------------------------------


def test_cost_tracker_caps_and_summary():
    tracker = CostTracker(max_cost_usd=0.001, max_requests=5)
    tracker.check_budget()
    cost = tracker.record("openai/gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    assert cost > 0
    assert tracker.requests == 1
    assert tracker.summary()["requests"] == 1

    # Exceed cost cap
    tracker.record("anthropic/claude-3.5-sonnet", prompt_tokens=1000, completion_tokens=1000)
    with pytest.raises(BudgetExceededError):
        tracker.check_budget()

    # Request cap
    req_tracker = CostTracker(max_requests=1)
    req_tracker.record("openai/gpt-4o-mini", 10, 10)
    with pytest.raises(BudgetExceededError):
        req_tracker.check_budget()


def test_llm_client_rate_limit_backoff_and_secret_hygiene():
    attempts = {"count": 0}
    sleeps: list[float] = []
    secret_key = "sk-test-super-secret-do-not-leak-999"

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        assert request.headers["Authorization"] == f"Bearer {secret_key}"
        if attempts["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"}, json={"error": "rate_limited"})
        if attempts["count"] == 2:
            return httpx.Response(503, json={"error": "upstream_unavailable"})
        return httpx.Response(200, json=_mock_chat_response("Recovered cleanly after retries."))

    transport = httpx.MockTransport(handler)
    with LLMClient(
        provider="openrouter",
        api_key=secret_key,
        transport=transport,
        max_retries=4,
        backoff_base_seconds=0.1,
        sleep_fn=sleeps.append,
    ) as client:
        assert secret_key not in repr(client)
        resp = client.complete("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}])
        assert resp.content == "Recovered cleanly after retries."
        assert attempts["count"] == 3
        assert len(sleeps) == 2
        assert sleeps[0] == pytest.approx(0.25)

    # Verify non-retryable 401 error never leaks the API key.
    def fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized")

    with LLMClient(
        provider="openrouter",
        api_key=secret_key,
        transport=httpx.MockTransport(fail_handler),
        max_retries=1,
        sleep_fn=lambda _: None,
    ) as fail_client:
        with pytest.raises(RuntimeError) as exc_info:
            fail_client.complete("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}])
        assert secret_key not in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Resumable corpus generation & mocked API distillation
# ---------------------------------------------------------------------------


def test_generate_real_corpus_resumable_checkpointing(tmp_path: Path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        user_prompt = body["messages"][-1]["content"]
        calls.append(user_prompt)
        return httpx.Response(
            200,
            json=_mock_chat_response(f"Generated response #{len(calls)} for: {user_prompt[:40]}"),
        )

    # First run: cap at 2 requests out of 4 total jobs.
    tracker1 = CostTracker(max_requests=2)
    with LLMClient(
        provider="openrouter",
        api_key="sk-mock",
        transport=httpx.MockTransport(handler),
        cost_tracker=tracker1,
        sleep_fn=lambda _: None,
    ) as client1:
        first_batch = generate_real_corpus(
            output_dir=tmp_path,
            models=["openai/gpt-4o-mini"],
            domains=["coding"],
            samples_per_model=4,
            resume=True,
            max_workers=1,
            seed=42,
            llm_client=client1,
        )

    assert len(first_batch) == 2
    assert len(calls) == 2
    assert len(load_jsonl(tmp_path / "raw_corpus.jsonl")) == 2

    # Second run with resume=True: must skip the first 2 and only request the remaining 2.
    tracker2 = CostTracker(max_requests=10)
    with LLMClient(
        provider="openrouter",
        api_key="sk-mock",
        transport=httpx.MockTransport(handler),
        cost_tracker=tracker2,
        sleep_fn=lambda _: None,
    ) as client2:
        second_batch = generate_real_corpus(
            output_dir=tmp_path,
            models=["openai/gpt-4o-mini"],
            domains=["coding"],
            samples_per_model=4,
            resume=True,
            max_workers=1,
            seed=42,
            llm_client=client2,
        )

    assert len(second_batch) == 4
    assert len(calls) == 4
    assert len({s.sample_id for s in second_batch}) == 4


def test_distill_training_pairs_hard_negatives_identity_and_resume(tmp_path: Path):
    samples = [
        DatasetSample(
            sample_id="code_sample_1",
            domain="coding",
            text="Right now `is_file()` swallows that error and returns `False` in `src/cache/store.py`.",
            source_model="openai/gpt-4o-mini",
            split="train",
            is_out_of_lexicon=False,
        ),
        DatasetSample(
            sample_id="bad_code_sample",
            domain="coding",
            text="Update `parse_header()` in `net/frames.py` to raise on malformed bytes.",
            source_model="openai/gpt-4o-mini",
            split="train",
            is_out_of_lexicon=True,
        ),
    ]
    identity_samples = [
        DatasetSample(
            sample_id="clear_human_1",
            domain="human_control",
            text="Keep pull requests small enough that a teammate can review them in fifteen minutes.",
            source_model="human",
            split="train",
            is_out_of_lexicon=True,
        )
    ]

    api_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        user_text = body["messages"][-1]["content"]
        api_calls.append(user_text)

        if "is_file()" in user_text:
            payload = {
                "needs_edit": True,
                "chosen": "Right now, `is_file()` ignores that error and returns `False` in `src/cache/store.py`.",
                "over_edited": "The file check fails completely and breaks the entire storage module.",
                "clunky_literal": "At the present moment `is_file()` does not handle that error and yields `False` within `src/cache/store.py`.",
            }
        elif "parse_header()" in user_text:
            # Simulate a bad teacher rewrite that drops protected code spans.
            payload = {
                "needs_edit": True,
                "chosen": "Update the header parser in the network module to raise on bad bytes.",
                "over_edited": "Rewrite the whole network stack.",
                "clunky_literal": "Make an update to the header parser.",
            }
        else:
            # Identity/no-op contrast call
            payload = {
                "over_edited": "Ensure all code review submissions remain strictly diminutive in scope to facilitate rapid peer evaluation.",
            }
        return httpx.Response(200, json=_mock_chat_response(json.dumps(payload)))

    with LLMClient(
        provider="openrouter",
        api_key="sk-mock",
        transport=httpx.MockTransport(handler),
        sleep_fn=lambda _: None,
    ) as client:
        pairs = distill_training_pairs(
            samples,
            identity_samples=identity_samples,
            output_dir=tmp_path,
            teacher_model="anthropic/claude-3.5-sonnet",
            resume=True,
            max_workers=1,
            min_code_preservation=1.0,
            llm_client=client,
        )

    # bad_code_sample dropped because code_preservation_rate < 1.0; 2 valid pairs remain.
    assert len(pairs) == 2
    by_id = {p.sample_id: p for p in pairs}

    code_pair = by_id["code_sample_1"]
    assert code_pair.is_identity is False
    assert code_pair.code_preservation_rate == 1.0
    assert "`is_file()`" in code_pair.chosen
    assert code_pair.rejected != code_pair.chosen
    neg_types = {n["type"] for n in code_pair.hard_negatives}
    assert {"over_edited", "clunky_literal", "dense_original"} <= neg_types

    id_pair = by_id["clear_human_1"]
    assert id_pair.is_identity is True
    assert id_pair.chosen == id_pair.source
    assert id_pair.rejected and id_pair.rejected != id_pair.source

    # Calling distill again with resume=True on already-completed pairs makes zero new API calls
    # for completed IDs.
    prev_calls = len(api_calls)
    with LLMClient(
        provider="openrouter",
        api_key="sk-mock",
        transport=httpx.MockTransport(handler),
        sleep_fn=lambda _: None,
    ) as client2:
        resumed = distill_training_pairs(
            [samples[0]],
            identity_samples=identity_samples,
            output_dir=tmp_path,
            teacher_model="anthropic/claude-3.5-sonnet",
            resume=True,
            max_workers=1,
            llm_client=client2,
        )
    assert len(resumed) == 2
    assert len(api_calls) == prev_calls


def test_cli_arg_parser_smoke():
    parser = build_arg_parser()
    gen_args = parser.parse_args(
        ["generate", "--models", "openai/gpt-4o-mini", "--max-cost-usd", "2.5", "--no-resume"]
    )
    assert gen_args.cmd == "generate"
    assert gen_args.models == ["openai/gpt-4o-mini"]
    assert gen_args.max_cost_usd == pytest.approx(2.5)
    assert gen_args.resume is False

    split_args = parser.parse_args(["split", "--val-ratio", "0.1", "--test-ratio", "0.2"])
    assert split_args.cmd == "split"
    assert split_args.val_ratio == pytest.approx(0.1)

    dist_args = parser.parse_args(["distill", "--teacher-model", "openai/gpt-4o", "--max-requests", "50"])
    assert dist_args.cmd == "distill"
    assert dist_args.teacher_model == "openai/gpt-4o"
    assert dist_args.max_requests == 50
