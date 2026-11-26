"""Tests for the lazy Python SDK (rewriter.py), CLI (cli.py), and HTTP server (server.py)."""

from __future__ import annotations

import io
import json
import sys
import threading
import types
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest

import psycho_rev
from psycho_rev import cli
from psycho_rev.rewriter import RewriteResult, Rewriter, unified_diff
from psycho_rev.server import create_server


@dataclass
class _StubReport:
    word_count: int = 8
    sentence_count: int = 1
    unattested_bigram_rate: float = 0.25
    trigram_perplexity: float = 120.0
    mean_dep_length: float = 2.4
    std_dep_length: float = 0.9
    reduced_orc_count: int = 1
    full_orc_count: int = 0
    reduced_orc_per_100w: float = 12.5
    full_orc_per_100w: float = 0.0
    stacked_noun_count: int = 0
    stacked_noun_per_100w: float = 0.0
    denominal_verb_count: int = 1
    denominal_verb_per_100w: float = 12.5
    inanimate_subj_count: int = 0
    inanimate_subj_per_100w: float = 0.0
    difficulty_score: float = 24.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _StubAnalyzer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def analyze(self, text: str) -> _StubReport:
        self.calls.append(text)
        score = 12.0 if "clear" in text.lower() or "use " in text.lower() else 28.5
        return _StubReport(word_count=len(text.split()), difficulty_score=score)


class _StubReverser:
    def __init__(self, *, skip_clear_marker: str = "ALREADY CLEAR") -> None:
        self.calls: list[tuple[str, bool]] = []
        self.last_skipped_as_clear: bool = False
        self.skip_clear_marker = skip_clear_marker
        self.closed = False

    def rewrite(self, text: str, skip_if_clear: bool = True) -> str:
        self.calls.append((text, skip_if_clear))
        if skip_if_clear and self.skip_clear_marker in text:
            self.last_skipped_as_clear = True
            return text
        self.last_skipped_as_clear = False
        return text.replace("leverage", "use").replace("workshopped", "discussed")

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Package lazy exports & version
# ---------------------------------------------------------------------------


def test_package_exports_and_version():
    assert psycho_rev.__version__ == "0.2.0"
    assert psycho_rev.Rewriter is Rewriter
    assert psycho_rev.RewriteResult is RewriteResult
    with pytest.raises(AttributeError):
        _ = getattr(psycho_rev, "NonExistentSymbol")


# ---------------------------------------------------------------------------
# unified_diff helper
# ---------------------------------------------------------------------------


def test_unified_diff_changed_and_unchanged():
    assert unified_diff("Same text.", "Same text.") == ""
    diff = unified_diff(
        "We should leverage `Pool.acquire()` now.\n",
        "We should use `Pool.acquire()` now.\n",
        filename="notes.md",
    )
    assert "--- a/notes.md" in diff
    assert "+++ b/notes.md" in diff
    assert "-We should leverage `Pool.acquire()` now." in diff
    assert "+We should use `Pool.acquire()` now." in diff


# ---------------------------------------------------------------------------
# Rewriter SDK
# ---------------------------------------------------------------------------


def test_rewriter_rewrite_and_code_preservation():
    analyzer = _StubAnalyzer()
    stub_rev = _StubReverser()
    rw = Rewriter(backend="classical", analyzer=analyzer, reverser=stub_rev)

    text = "We should leverage `Pool.acquire()` in src/cache/store.py."
    res = rw.rewrite(text)

    assert isinstance(res, RewriteResult)
    assert res.original == text
    assert res.rewritten == "We should use `Pool.acquire()` in src/cache/store.py."
    assert res.changed is True
    assert res.skipped_as_clear is False
    assert res.backend == "classical"
    assert res.latency_ms >= 0.0
    assert res.code_preservation == 1.0
    assert res.before_difficulty == 28.5
    assert res.after_difficulty == 12.0
    assert res.to_dict()["rewritten"] == res.rewritten


def test_rewriter_skip_if_clear_and_batch():
    analyzer = _StubAnalyzer()
    stub_rev = _StubReverser()
    rw = Rewriter(backend="classical", analyzer=analyzer, reverser=stub_rev)

    results = rw.rewrite_batch(["ALREADY CLEAR sentence.", "We workshopped the plan."])
    assert len(results) == 2
    assert results[0].changed is False
    assert results[0].skipped_as_clear is True
    assert results[1].changed is True
    assert results[1].rewritten == "We discussed the plan."

    # Explicitly disable skip_if_clear per call
    res_no_skip = rw.rewrite("ALREADY CLEAR leverage test.", skip_if_clear=False)
    assert res_no_skip.changed is True
    assert res_no_skip.skipped_as_clear is False


def test_rewriter_rewrite_file_read_only_and_in_place(tmp_path: Path):
    target = tmp_path / "doc.md"
    original = "Please leverage `parse_header()` here.\n"
    target.write_text(original, encoding="utf-8")

    rw = Rewriter(backend="classical", analyzer=_StubAnalyzer(), reverser=_StubReverser())
    res_ro = rw.rewrite_file(target, in_place=False)
    assert res_ro.changed is True
    assert target.read_text(encoding="utf-8") == original

    res_ip = rw.rewrite_file(target, in_place=True)
    assert res_ip.changed is True
    assert target.read_text(encoding="utf-8") == "Please use `parse_header()` here.\n"


def test_rewriter_openai_and_openrouter_backend_wiring(monkeypatch: pytest.MonkeyPatch):
    captured: list[dict[str, Any]] = []

    class FakeOpenAICompatibleReverser:
        def __init__(
            self,
            api_key: str,
            base_url: str,
            model: str | None = None,
            skip_if_clear: bool = True,
        ) -> None:
            captured.append(
                {
                    "api_key": api_key,
                    "base_url": base_url,
                    "model": model,
                    "skip_if_clear": skip_if_clear,
                }
            )

        def rewrite(self, text: str) -> str:
            return text.replace("leverage", "use")

        def close(self) -> None:
            pass

    import psycho_rev
    fake_neural = types.ModuleType("psycho_rev.neural")
    fake_neural.OpenAICompatibleReverser = FakeOpenAICompatibleReverser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycho_rev.neural", fake_neural)
    monkeypatch.setattr(psycho_rev, "neural", fake_neural, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

    with Rewriter(
        backend="openrouter",
        model="qwen/qwen-2.5-72b-instruct",
        compute_metrics=False,
    ) as rw:
        res = rw.rewrite("We leverage tools.")

    assert res.rewritten == "We use tools."
    assert res.backend == "openrouter"
    assert res.before_difficulty is None
    assert captured[0]["api_key"] == "sk-or-test-key"
    assert captured[0]["base_url"] == "https://openrouter.ai/api/v1"
    assert captured[0]["model"] == "qwen/qwen-2.5-72b-instruct"


def test_rewriter_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch):
    fake_neural = types.ModuleType("psycho_rev.neural")

    class FakeAPI:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

    fake_neural.OpenAICompatibleReverser = FakeAPI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycho_rev.neural", fake_neural)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    rw = Rewriter(backend="openai", compute_metrics=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        rw.rewrite("Some text to rewrite.")


def test_rewriter_unsupported_backend_raises():
    with pytest.raises(ValueError, match="Unsupported backend"):
        Rewriter(backend="unknown-backend")


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------


def _patch_cli_rewriter(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    init_calls: list[dict[str, Any]] = []

    class PatchedRewriter(Rewriter):
        def __init__(self, backend: str = "classical", **kwargs: Any) -> None:
            init_calls.append({"backend": backend, **kwargs})
            kwargs.setdefault("analyzer", _StubAnalyzer())
            kwargs.setdefault("reverser", _StubReverser())
            super().__init__(backend=backend, **kwargs)

    monkeypatch.setattr("psycho_rev.rewriter.Rewriter", PatchedRewriter)
    return init_calls


def test_cli_analyze_positional_and_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    _patch_cli_rewriter(monkeypatch)
    rc = cli.main(["analyze", "We leverage the tool.", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["difficulty_score"] == 28.5

    rc2 = cli.main(["analyze", "We leverage the tool."])
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert "Difficulty score:" in out2


def test_cli_rewrite_stdin_plain_and_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    init_calls = _patch_cli_rewriter(monkeypatch)

    monkeypatch.setattr(sys, "stdin", io.StringIO("We leverage pipes."))
    rc = cli.main(["rewrite", "-", "--plain", "--no-skip-clear", "--unsafe-classical"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "We use pipes."
    assert init_calls[-1]["skip_if_clear"] is False
    assert init_calls[-1]["safe_mode"] is False

    monkeypatch.setattr(sys, "stdin", io.StringIO("We workshopped the API."))
    rc2 = cli.main(["rewrite", "--json"])
    assert rc2 == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rewritten"] == "We discussed the API."
    assert payload["changed"] is True
    assert payload["before_difficulty"] == 28.5


def test_cli_rewrite_file_diff_and_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _patch_cli_rewriter(monkeypatch)
    doc = tmp_path / "draft.md"
    doc.write_text("Teams leverage caches.\n", encoding="utf-8")

    rc_diff = cli.main(["rewrite", "--file", str(doc), "--diff"])
    assert rc_diff == 0
    diff_out = capsys.readouterr().out
    assert f"--- a/{doc}" in diff_out
    assert "-Teams leverage caches." in diff_out
    assert "+Teams use caches." in diff_out
    assert doc.read_text(encoding="utf-8") == "Teams leverage caches.\n"

    rc_ip = cli.main(["rewrite", "--file", str(doc), "--in-place"])
    assert rc_ip == 0
    assert doc.read_text(encoding="utf-8") == "Teams use caches.\n"


def test_cli_rewrite_in_place_without_file_errors():
    with pytest.raises(SystemExit) as exc:
        cli.main(["rewrite", "inline text", "--in-place"])
    assert exc.value.code == 2


def test_cli_delegates_build_corpus_and_evaluate(monkeypatch: pytest.MonkeyPatch):
    seen: list[tuple[str, list[str]]] = []

    fake_cb = types.ModuleType("psycho_rev.corpus_builder")
    fake_cb.main = lambda: seen.append(("corpus_builder", list(sys.argv[1:])))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycho_rev.corpus_builder", fake_cb)

    fake_ev = types.ModuleType("psycho_rev.evaluate")
    fake_ev.main = lambda: seen.append(("evaluate", list(sys.argv[1:])))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycho_rev.evaluate", fake_ev)

    assert cli.main(["build-corpus", "--out", "data.jsonl"]) == 0
    assert cli.main(["evaluate", "--classical-only"]) == 0
    assert seen == [
        ("corpus_builder", ["--out", "data.jsonl"]),
        ("evaluate", ["--classical-only"]),
    ]


# ---------------------------------------------------------------------------
# Live loopback HTTP server (server.py)
# ---------------------------------------------------------------------------


def _http_request(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


def test_server_live_loopback_endpoints():
    stub_rev = _StubReverser()
    rewriter = Rewriter(backend="classical", analyzer=_StubAnalyzer(), reverser=stub_rev)
    server = create_server(rewriter, host="127.0.0.1", port=0, max_body_bytes=256)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # GET /health
        status, data = _http_request(f"{base}/health")
        assert status == 200
        assert data == {"status": "ok", "backend": "classical"}

        # POST /analyze
        status, data = _http_request(
            f"{base}/analyze",
            method="POST",
            body=json.dumps({"text": "We leverage caches."}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert status == 200
        assert data["difficulty_score"] == 28.5

        # POST /rewrite (default skip_if_clear=True)
        status, data = _http_request(
            f"{base}/rewrite",
            method="POST",
            body=json.dumps({"text": "ALREADY CLEAR leverage.", "skip_if_clear": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert status == 200
        assert data["skipped_as_clear"] is True
        assert data["changed"] is False

        # POST /rewrite (skip_if_clear=False)
        status, data = _http_request(
            f"{base}/rewrite",
            method="POST",
            body=json.dumps({"text": "ALREADY CLEAR leverage.", "skip_if_clear": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert status == 200
        assert data["changed"] is True
        assert data["rewritten"] == "ALREADY CLEAR use."

        # 400 on malformed JSON
        status, data = _http_request(
            f"{base}/rewrite",
            method="POST",
            body=b"{not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert "error" in data

        # 400 on missing text field
        status, data = _http_request(
            f"{base}/analyze",
            method="POST",
            body=json.dumps({"wrong": 123}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert status == 400

        # 400 on oversized payload
        status, data = _http_request(
            f"{base}/rewrite",
            method="POST",
            body=json.dumps({"text": "x" * 512}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert status == 400

        # 405 on wrong method
        status, _ = _http_request(f"{base}/rewrite", method="GET")
        assert status == 405
        status, _ = _http_request(f"{base}/health", method="POST", body=b"{}")
        assert status == 405

        # 404 on unknown path
        status, _ = _http_request(f"{base}/nonexistent", method="GET")
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_server_returns_500_on_backend_error():
    class ExplodingReverser:
        def rewrite(self, text: str) -> str:
            raise RuntimeError("boom")

    rewriter = Rewriter(
        backend="classical",
        analyzer=_StubAnalyzer(),
        reverser=ExplodingReverser(),
    )
    server = create_server(rewriter, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, data = _http_request(
            f"http://127.0.0.1:{port}/rewrite",
            method="POST",
            body=json.dumps({"text": "trigger failure"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert status == 500
        assert "boom" in data["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Regression tests: skip_if_clear override, OpenRouter default, cleanup, lock
# ---------------------------------------------------------------------------


def test_openai_reverser_skip_if_clear_override_and_last_skipped_flag():
    import httpx
    from psycho_rev.neural import OpenAICompatibleReverser

    requests_made: list[dict[str, Any]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        requests_made.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Rewritten clear text."}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    rev = OpenAICompatibleReverser(
        api_key="test-key",
        skip_if_clear=True,
        clear_difficulty_threshold=20.0,
        analyzer=_StubAnalyzer(),
        client=client,
    )
    try:
        # Clear text (difficulty 12.0 <= 20.0) skipped by default
        out_skipped = rev.rewrite("This is clear text.")
        assert out_skipped == "This is clear text."
        assert rev.last_skipped_as_clear is True
        assert len(requests_made) == 0

        # Explicit per-call override forces rewrite
        out_forced = rev.rewrite("This is clear text.", skip_if_clear=False)
        assert out_forced == "Rewritten clear text."
        assert rev.last_skipped_as_clear is False
        assert len(requests_made) == 1
    finally:
        client.close()


def test_hybrid_reverser_skips_classical_when_neural_skips_as_clear():
    from psycho_rev.neural import HybridReverser

    class _RecordingClassical:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def rewrite(self, text: str) -> str:
            self.calls.append(text)
            return text + " [classical]"

    neural_stub = _StubReverser()
    classical_stub = _RecordingClassical()
    hybrid = HybridReverser(neural_stub, classical=classical_stub, skip_if_clear=True)

    # Neural skips -> classical must NOT run and last_skipped_as_clear propagates
    out_skip = hybrid.rewrite("ALREADY CLEAR leverage.")
    assert out_skip == "ALREADY CLEAR leverage."
    assert hybrid.last_skipped_as_clear is True
    assert classical_stub.calls == []

    # Per-call override skip_if_clear=False -> both neural and classical run
    out_force = hybrid.rewrite("ALREADY CLEAR leverage.", skip_if_clear=False)
    assert out_force == "ALREADY CLEAR use. [classical]"
    assert hybrid.last_skipped_as_clear is False
    assert classical_stub.calls == ["ALREADY CLEAR use."]


def test_best_of_n_skip_if_clear_override_and_flag():
    from psycho_rev.neural import BestOfNPsycholinguisticReverser

    class _FakeGen:
        def __init__(self) -> None:
            self.calls: int = 0

        def generate_candidates(self, text: str, **kwargs: Any) -> list[str]:
            self.calls += 1
            return [text.replace("clear", "simple")]

    @dataclass
    class _FakeSemReport:
        cosine_similarity: float = 0.95
        nli_min_entailment: float = 0.9
        nli_max_contradiction: float = 0.02
        code_preservation: float = 1.0
        gpt2_perplexity: float = 20.0
        length_ratio: float = 1.0
        acceptability_min: float = 0.95

    class _FakeSem:
        def evaluate_batch(self, originals: list[str], candidates: list[str]) -> list[_FakeSemReport]:
            return [_FakeSemReport() for _ in candidates]

    gen = _FakeGen()
    bon = BestOfNPsycholinguisticReverser(
        gen,
        _StubAnalyzer(),  # type: ignore[arg-type]
        _FakeSem(),  # type: ignore[arg-type]
        skip_if_clear=True,
        clear_difficulty_threshold=20.0,
    )

    out_skip = bon.rewrite("This is clear.")
    assert out_skip == "This is clear."
    assert bon.last_skipped_as_clear is True
    assert gen.calls == 0

    out_force = bon.rewrite("This is clear.", skip_if_clear=False)
    assert out_force == "This is simple."
    assert bon.last_skipped_as_clear is False
    assert gen.calls == 1


def test_rewriter_openrouter_default_model(monkeypatch: pytest.MonkeyPatch):
    captured: list[dict[str, Any]] = []

    class FakeOpenAICompatibleReverser:
        def __init__(
            self,
            api_key: str,
            base_url: str,
            model: str = "gpt-4o-mini",
            skip_if_clear: bool = True,
        ) -> None:
            captured.append({"base_url": base_url, "model": model})

        def rewrite(self, text: str, *, skip_if_clear: bool | None = None) -> str:
            return text

        def close(self) -> None:
            pass

    fake_neural = types.ModuleType("psycho_rev.neural")
    fake_neural.OpenAICompatibleReverser = FakeOpenAICompatibleReverser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycho_rev.neural", fake_neural)
    monkeypatch.setattr(psycho_rev, "neural", fake_neural, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

    with Rewriter(backend="openrouter", compute_metrics=False) as rw_or:
        rw_or.rewrite("Sample text.")
    with Rewriter(backend="openai", compute_metrics=False) as rw_oai:
        rw_oai.rewrite("Sample text.")

    assert captured[0]["model"] == "openai/gpt-4o-mini"
    assert captured[1]["model"] == "gpt-4o-mini"


@pytest.mark.parametrize("backend", ["best-of-n", "hybrid-bon"])
def test_rewriter_semantic_evaluator_rerank_role_and_cleanup(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    gen_instances: list[Any] = []
    sem_instances: list[Any] = []

    class FakeGen:
        def __init__(self, model_name: str = "") -> None:
            self.model_name = model_name
            self.closed = False
            gen_instances.append(self)

        def close(self) -> None:
            self.closed = True

    class FakeSem:
        def __init__(self, role: str = "eval") -> None:
            self.role = role
            self.closed = False
            sem_instances.append(self)

        def close(self) -> None:
            self.closed = True

    class FakeBoN:
        def __init__(self, gen: Any, analyzer: Any, sem: Any, **kwargs: Any) -> None:
            self.last_skipped_as_clear = False

        def rewrite(self, text: str, *, skip_if_clear: bool | None = None) -> str:
            return text.replace("leverage", "use")

    class FakeHybrid:
        def __init__(self, neural: Any, **kwargs: Any) -> None:
            self.neural = neural
            self.last_skipped_as_clear = False

        def rewrite(self, text: str, *, skip_if_clear: bool | None = None) -> str:
            return self.neural.rewrite(text, skip_if_clear=skip_if_clear)

    fake_neural = types.ModuleType("psycho_rev.neural")
    fake_neural.QwenZeroShotReverser = FakeGen  # type: ignore[attr-defined]
    fake_neural.BestOfNPsycholinguisticReverser = FakeBoN  # type: ignore[attr-defined]
    fake_neural.HybridReverser = FakeHybrid  # type: ignore[attr-defined]

    fake_semantics = types.ModuleType("psycho_rev.semantics")
    fake_semantics.SemanticEvaluator = FakeSem  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "psycho_rev.neural", fake_neural)
    monkeypatch.setitem(sys.modules, "psycho_rev.semantics", fake_semantics)

    with Rewriter(backend=backend, analyzer=_StubAnalyzer(), compute_metrics=False) as rw:
        res = rw.rewrite("We leverage tools.")
        assert res.rewritten == "We use tools."
        assert len(sem_instances) == 1
        assert sem_instances[0].role == "rerank"
        assert sem_instances[0].closed is False
        assert gen_instances[0].closed is False

    assert sem_instances[0].closed is True
    assert gen_instances[0].closed is True


def test_server_serializes_concurrent_rewriter_calls():
    import time

    active = 0
    max_active = 0
    state_lock = threading.Lock()

    class _SlowReverser:
        def rewrite(self, text: str, *, skip_if_clear: bool | None = None) -> str:
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.08)
            with state_lock:
                active -= 1
            return text.replace("leverage", "use")

    rewriter = Rewriter(
        backend="classical",
        analyzer=_StubAnalyzer(),
        reverser=_SlowReverser(),
        compute_metrics=False,
    )
    server = create_server(rewriter, host="127.0.0.1", port=0)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"

    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    try:
        results: list[tuple[int, dict[str, Any]]] = []

        def _worker() -> None:
            res = _http_request(
                f"{base}/rewrite",
                method="POST",
                body=json.dumps({"text": "We leverage caches."}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            results.append(res)

        t1 = threading.Thread(target=_worker)
        t2 = threading.Thread(target=_worker)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert len(results) == 2
        assert all(status == 200 for status, _ in results)
        assert max_active == 1
    finally:
        server.shutdown()
        server.server_close()
        srv_thread.join(timeout=5)
