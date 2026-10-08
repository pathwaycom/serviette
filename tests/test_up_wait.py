"""Unit tests for the server-readiness wait in ``serviette up`` (no Pathway).

``up`` must announce the URL only once the server answers its health check:
uvicorn binds the port after the lifespan warm-up, and a local embedder's
warm-up (torch import + model load) keeps the port closed for tens of
seconds — a URL printed at spawn time sends the user to "connection refused".
"""

from __future__ import annotations

import logging

import pytest

from serviette import up
from serviette.config.schema import ServietteConfig
from serviette.up import _probe_host, _server_url, _wait_for_index, _wait_for_server


class FakeProc:
    pid = 4242

    def __init__(self, codes):
        # Successive poll() results; the last one repeats.
        self._codes = list(codes)

    def poll(self):
        if len(self._codes) > 1:
            return self._codes.pop(0)
        return self._codes[0]


def _config(
    host: str = "127.0.0.1", port: int = 8989, index_wait_timeout: float | None = 120.0
) -> ServietteConfig:
    return ServietteConfig.model_validate(
        {
            "sources": [{"type": "fs", "path": "."}],
            "vector_db": {"type": "duckdb", "path": "e.duckdb"},
            "embedder": {"type": "mock"},
            "server": {"host": host, "port": port},
            "up": {"index_wait_timeout": index_wait_timeout},
        }
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("serviette.up.time.sleep", lambda _s: None)


def test_waits_until_health_answers():
    probes = iter([False, False, False, True])
    calls = 0

    def ready(_config):
        nonlocal calls
        calls += 1
        return next(probes)

    result = _wait_for_server(
        _config(), FakeProc([None]), FakeProc([None]), ready=ready
    )
    assert result is None
    assert calls == 4


def test_reports_server_death_before_ready():
    # Server dies (e.g. port in use -> uvicorn exits 3) while still unhealthy.
    result = _wait_for_server(
        _config(), FakeProc([None, 3]), FakeProc([None]), ready=lambda _c: False
    )
    assert result == 3


def test_reports_indexer_failure_before_ready():
    result = _wait_for_server(
        _config(), FakeProc([None]), FakeProc([None, 1]), ready=lambda _c: False
    )
    assert result == 1


def test_indexer_static_exit_zero_is_not_a_failure():
    # A one-shot (static) indexer finishing is fine; keep waiting for the server.
    probes = iter([False, True])
    result = _wait_for_server(
        _config(), FakeProc([None]), FakeProc([0]), ready=lambda _c: next(probes)
    )
    assert result is None


def test_stop_request_cuts_the_wait_short():
    result = _wait_for_server(
        _config(),
        FakeProc([None]),
        FakeProc([None]),
        should_stop=lambda: True,
        ready=lambda _c: False,
    )
    assert result is None


def test_server_url_maps_wildcard_binds_to_localhost():
    assert _server_url(_config("0.0.0.0", 8989)) == "http://localhost:8989"
    assert _server_url(_config("127.0.0.1", 8989)) == "http://localhost:8989"
    assert _server_url(_config("192.168.1.5", 9000)) == "http://192.168.1.5:9000"


@pytest.mark.parametrize(
    ("bind", "probe"),
    [
        ("0.0.0.0", "127.0.0.1"),
        ("", "127.0.0.1"),
        ("::", "[::1]"),
        ("127.0.0.1", "127.0.0.1"),
        # A specific address or hostname is reachable only as itself: probing
        # loopback for it never answers and up used to wait forever.
        ("192.168.1.5", "192.168.1.5"),
        ("rag.internal", "rag.internal"),
        ("fe80::1", "[fe80::1]"),
    ],
)
def test_readiness_probe_follows_the_bind_address(bind, probe):
    assert _probe_host(bind) == probe


def test_index_wait_stops_on_shutdown_request(monkeypatch):
    """A SIGTERM during the first indexing pass must end the wait even though
    the store never becomes ready (the caller then tears the children down)."""

    monkeypatch.setattr("serviette.up._sources_look_empty", lambda _c: False)
    result = _wait_for_index(
        _config(),
        FakeProc([None]),
        should_stop=lambda: True,
        ready=lambda _c, allow_empty: False,
    )
    assert result is None


def test_index_wait_reports_indexer_failure(monkeypatch):
    monkeypatch.setattr("serviette.up._sources_look_empty", lambda _c: False)
    result = _wait_for_index(
        _config(), FakeProc([3]), ready=lambda _c, allow_empty: False
    )
    assert result == 3


def _fake_clock(monkeypatch, step: float = 30.0):
    """``time.monotonic`` advancing ``step`` seconds per call."""

    ticks = iter(i * step for i in range(10_000))
    monkeypatch.setattr("serviette.up.time.monotonic", lambda: next(ticks))


def test_index_wait_gives_up_after_the_timeout(monkeypatch, caplog):
    """A non-empty folder whose documents never yield a chunk (all skipped or
    failing to parse) must not hold the server back forever."""

    monkeypatch.setattr("serviette.up._sources_look_empty", lambda _c: False)
    _fake_clock(monkeypatch)
    probes = 0

    def never_ready(_config, allow_empty):
        nonlocal probes
        probes += 1
        return False

    result = _wait_for_index(
        _config(index_wait_timeout=100.0), FakeProc([None]), ready=never_ready
    )
    assert result is None
    assert probes <= 6  # ~100s at 30s per probe, not thousands
    assert "starting the server over an empty index" in caplog.text


def test_index_wait_timeout_none_waits_indefinitely(monkeypatch, caplog):
    monkeypatch.setattr("serviette.up._sources_look_empty", lambda _c: False)
    _fake_clock(monkeypatch)
    probes = 0

    def ready(_config, allow_empty):
        nonlocal probes
        probes += 1
        return probes >= 50  # far beyond the default 120s at 30s per probe

    result = _wait_for_index(
        _config(index_wait_timeout=None), FakeProc([None]), ready=ready
    )
    assert result is None
    assert probes == 50
    assert "starting the server over an empty index" not in caplog.text


def test_index_wait_ready_before_timeout_is_silent(monkeypatch, caplog):
    monkeypatch.setattr("serviette.up._sources_look_empty", lambda _c: False)
    _fake_clock(monkeypatch)
    probes = iter([False, True])
    result = _wait_for_index(
        _config(index_wait_timeout=100.0),
        FakeProc([None]),
        ready=lambda _c, allow_empty: next(probes),
    )
    assert result is None
    assert "empty index" not in caplog.text


def test_fingerprint_is_confirmed_by_up_before_the_indexer_starts(tmp_path, monkeypatch):
    """The fingerprint question is up's: asked in a quiet terminal before the
    indexer is spawned (a child's prompt would sit under the heartbeat), and
    the child is told the answer is already given."""

    events: list = []

    def fake_check(config):
        events.append("check")

    monkeypatch.setattr("serviette.indexer.fingerprint.check_fingerprint", fake_check)

    def fake_spawn(command, config_path, *, env=None, verbose=False):
        events.append((command, env))
        # indexer keeps running; the server "exits 0" so run() returns.
        return FakeProc([None]) if command == "indexer" else FakeProc([0])

    monkeypatch.setattr(up, "_spawn", fake_spawn)
    monkeypatch.setattr(up, "_wait_for_index", lambda *a, **k: None)
    monkeypatch.setattr(up, "_wait_for_server", lambda *a, **k: None)
    monkeypatch.setattr(up, "_terminate", lambda proc, name: None)
    monkeypatch.delenv("SERVIETTE_ACCEPT_FINGERPRINT_CHANGES", raising=False)

    config = ServietteConfig.model_validate(
        {
            "sources": [{"type": "fs", "path": str(tmp_path), "mode": "static"}],
            "vector_db": {"type": "duckdb", "path": str(tmp_path / "e.duckdb")},
            "embedder": {"type": "mock"},
            "workdir_path": str(tmp_path / "workdir"),
        }
    )
    assert up.run(config, "config.yaml") == 0
    assert events[0] == "check"
    command, env = events[1]
    assert command == "indexer"
    assert env["SERVIETTE_ACCEPT_FINGERPRINT_CHANGES"] == "1"
    # The server child inherits the plain environment.
    assert events[2][0] == "server" and events[2][1] is None


def test_up_refuses_when_the_fingerprint_is_declined(tmp_path, monkeypatch):
    spawned = []
    monkeypatch.setattr(up, "_spawn", lambda *a, **k: spawned.append(a))

    def refuse(config):
        raise SystemExit("Refusing to start: the configuration change above ...")

    monkeypatch.setattr("serviette.indexer.fingerprint.check_fingerprint", refuse)
    config = ServietteConfig.model_validate(
        {
            "sources": [{"type": "fs", "path": str(tmp_path), "mode": "static"}],
            "vector_db": {"type": "duckdb", "path": str(tmp_path / "e.duckdb")},
            "embedder": {"type": "mock"},
        }
    )
    with pytest.raises(SystemExit, match="Refusing to start"):
        up.run(config, "config.yaml")
    assert spawned == []  # nothing started


def test_server_probe_ignores_the_environment_proxy(monkeypatch):
    """A corporate http_proxy must not swallow the loopback health probe:
    the proxy refuses 127.0.0.1, and ``up`` would wait for a server that
    is already up."""
    import http.server
    import socketserver
    import threading

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, *_):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Health) as httpd:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            config = _config()
            config.server.port = httpd.server_address[1]
            # A proxy that accepts nothing: without trust_env=False the probe
            # would be sent there and fail.
            monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
            monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
            monkeypatch.delenv("no_proxy", raising=False)
            monkeypatch.delenv("NO_PROXY", raising=False)
            assert up._server_ready(config) is True
        finally:
            httpd.shutdown()


# ---------------------------------------------------------------------------
# Failing before the children start, and reporting a child's death
# ---------------------------------------------------------------------------


def _openai_config(tmp_path, **overrides):
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    data = {
        "sources": [{"type": "fs", "path": str(docs)}],
        "vector_db": {"type": "duckdb", "path": str(tmp_path / "e.duckdb")},
        "embedder": {"type": "openai"},
        **overrides,
    }
    return ServietteConfig.model_validate(data)


def test_up_refuses_openai_sections_without_a_key(tmp_path, monkeypatch):
    """No api_key in the config and no OPENAI_API_KEY: say so before any
    child starts, naming the sections, instead of the indexer dying on the
    SDK's traceback after "indexing in progress"."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    spawned: list = []
    monkeypatch.setattr(up, "_spawn", lambda *a, **k: spawned.append(a))
    cfg = _openai_config(tmp_path, llm={"type": "openai"})
    with pytest.raises(SystemExit) as exc:
        up.run(cfg, str(tmp_path / "config.yaml"))
    message = str(exc.value)
    assert "embedder (type: openai)" in message and "llm (type: openai)" in message
    assert "OPENAI_API_KEY" in message
    assert spawned == []

    # A key in either place satisfies the check.
    from serviette.config.schema import missing_openai_credentials

    assert missing_openai_credentials(
        _openai_config(tmp_path, embedder={"type": "openai", "api_key": "sk-x"}), roles=("embedder",)
    ) == []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert missing_openai_credentials(cfg, roles=("embedder", "llm")) == []
    # Non-OpenAI sections are never asked for it.
    monkeypatch.delenv("OPENAI_API_KEY")
    assert missing_openai_credentials(
        _openai_config(tmp_path, embedder={"type": "sentence_transformer"}), roles=("embedder", "llm")
    ) == []


@pytest.mark.parametrize(
    "tail, expected",
    [
        (["openai.OpenAIError: Missing credentials. Please pass an `api_key`"], "no OpenAI API key"),
        (["openai.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached"], "rate limit (HTTP 429)"),
        (["openai.RateLimitError: Error code: 429 - You exceeded your current quota"], "no remaining quota"),
        (["OSError: [Errno 98] error while attempting to bind on address ('127.0.0.1', 8989): address already in use"], "port is already taken"),
        (["milvus_lite.exceptions.DataDirLockedError: another process holds the lock"], "locked by another process"),
        (["pathway.engine.EngineError: error sending request for url (http://127.0.0.1:8080/v1/batch/objects)"], "could not be reached"),
        (["something nobody has seen before"], "last traceback above says why"),
        ([], "last traceback above says why"),
    ],
)
def test_failure_report_names_the_cause(tail, expected):
    report = up._failure_report("indexer", 1, tail)
    assert "serviette up STOPPED: the indexer exited with code 1." in report
    assert expected in report
    # Unmistakable: framed and preceded by a blank line, so it never reads
    # as one more INFO line from the engine.
    assert report.startswith("\n" + "=" * 72)
    assert report.rstrip().endswith("=" * 72)


def test_spawned_child_output_is_forwarded_and_kept(monkeypatch, capsys):
    """A child's output reaches the terminal line by line and its last lines
    stay available for the failure report."""

    import time

    real_popen = up.subprocess.Popen
    monkeypatch.setattr(up.subprocess, "Popen", lambda args, **kw: _real_popen(real_popen, args, **kw))
    proc = up._spawn("indexer", "config.yaml")  # _real_popen ignores the command
    code = proc.wait(timeout=30)
    for _ in range(100):
        if len(proc.tail) >= 3:
            break
        time.sleep(0.05)
    assert code == 3
    assert list(proc.tail)[-3:] == ["line one", "line two", "Traceback: boom"]
    assert "Traceback: boom" in capsys.readouterr().err


def _real_popen(popen, args, **kw):
    """Run a tiny script in place of `serviette <command>`: prints three
    lines (two to stdout, one to stderr) and exits 3."""

    script = (
        "import sys; print('line one'); print('line two');"
        " print('Traceback: boom', file=sys.stderr); sys.exit(3)"
    )
    return popen([args[0], "-c", script], **kw)


@pytest.mark.parametrize(
    "line, routine",
    [
        ("INFO:pathway_engine.persistence.input_snapshot:Persisting a chunk of 50 entries", True),
        ("INFO up: indexing in progress", True),
        ("INFO:     127.0.0.1:33800 - \"POST /api/v1/rag HTTP/1.1\" 200 OK", True),
        ("Batches: 100%|██████████| 1/1 [00:00<00:00, 24.93it/s]", True),
        ("", True),
        ("WARNING:pathway.internals.graph_runner:Received SIGTERM", False),
        ("ERROR:    [Errno 98] error while attempting to bind on address", False),
        ("Traceback (most recent call last):", False),
        ("  File \"/x/y.py\", line 3, in <module>", False),
        ("openai.OpenAIError: Missing credentials.", False),
        ("thread 'pathway:output_table-WeaviateWriter' panicked at src/engine/report_error.rs", False),
        ("INFORMATIVE line from a print()", False),  # a word, not the level prefix
    ],
)
def test_quiet_mode_hides_only_routine_lines(line, routine):
    assert up._is_routine(line) is routine


def test_quiet_mode_keeps_hidden_lines_for_the_failure_report(monkeypatch, capsys):
    """Hidden INFO lines still land in the tail (the diagnosis needs them);
    the error line reaches the terminal either way."""

    real_popen = up.subprocess.Popen
    script = (
        "import sys; print('INFO:engine:routine'); print('Traceback: boom', file=sys.stderr);"
        " sys.exit(3)"
    )
    monkeypatch.setattr(
        up.subprocess, "Popen", lambda args, **kw: real_popen([args[0], "-c", script], **kw)
    )
    proc = up._spawn("indexer", "config.yaml", verbose=False)
    proc.wait(timeout=30)
    import time

    for _ in range(100):
        if len(proc.tail) >= 2:
            break
        time.sleep(0.05)
    assert list(proc.tail) == ["INFO:engine:routine", "Traceback: boom"]
    err = capsys.readouterr().err
    assert "Traceback: boom" in err and "routine" not in err


# -- activity-based timeout ------------------------------------------------
#
# The timeout counts from the indexer's last sign of work, not from the start:
# a few large PDFs parse for minutes without producing a chunk, and a timeout
# counted from the start announced an empty index that was about to fill.


def test_activity_lines_are_recognised():
    act = up.Activity()
    t0 = act.last_seen
    for quiet in (
        "INFO:pathway_engine.connectors.monitoring:source_0: 0 entries (4 minibatch(es)) have been sent",
        "INFO:pathway_engine.persistence.input_snapshot:Persisting a chunk of 105 entries",
        "INFO:httpx:HTTP Request: GET http://127.0.0.1:8989/health",
        "",
    ):
        act.observe(quiet)
    assert act.last_seen == t0
    assert act.describe() == "waiting for the first documents"

    act.observe("INFO:serviette.indexer.graph:parsing /data/eu-1272-2008.pdf (18.3 MB)")
    assert act.last_seen > t0
    assert act.describe() == "parsing eu-1272-2008.pdf"
    act.observe("INFO:serviette.indexer.graph:parsed /data/eu-1272-2008.pdf: 412000 chars in 47.2s")
    assert act.describe() == "1 documents parsed"
    act.observe("INFO:serviette.indexer.graph:parsing /data/b.pdf (2.0 MB)")
    assert act.describe() == "parsing b.pdf (1 parsed so far)"

    for busy in (
        "INFO:pathway.xpacks.llm.parsers:PypdfParser starting to parse a document of length: 312",
        "INFO:pathway_engine.connectors.monitoring:source_0: 4 entries (2 minibatch(es)) have been sent",
    ):
        before = act.last_seen
        act.observe(busy)
        assert act.last_seen >= before
    act.observe("INFO:serviette.indexer.graph:parsed /data/b.pdf: 10 chars in 0.1s")
    act.observe("Batches: 100%|██████████| 1/1 [00:00<00:00,  6.13it/s]")
    assert act.describe() == "embedding (2 parsed so far)"


def test_index_wait_timeout_counts_from_the_last_activity(monkeypatch, caplog):
    """Steady parsing activity keeps the wait alive past the timeout; the
    warning fires only once the indexer has been silent for that long."""

    monkeypatch.setattr("serviette.up._sources_look_empty", lambda _c: False)
    caplog.set_level(logging.INFO, logger="serviette.up")
    _fake_clock(monkeypatch, step=10.0)
    proc = FakeProc([None])
    proc.activity = up.Activity()
    probes = 0

    def ready(_config, allow_empty):
        nonlocal probes
        probes += 1
        # Busy for the first ~400s (far past the 100s timeout), then silent.
        if probes <= 40:
            proc.activity.observe(f"INFO:serviette.indexer.graph:parsing /d/{probes}.pdf (1.0 MB)")
        return False

    result = _wait_for_index(_config(index_wait_timeout=100.0), proc, ready=ready)
    assert result is None
    assert probes > 40, "the timeout fired while the indexer was still busy"
    assert "showed no activity for" in caplog.text
    assert "parsing 40.pdf" in caplog.text  # the heartbeat says what is going on


def test_index_wait_without_activity_tracking_counts_from_the_start(monkeypatch, caplog):
    """A process without ``.activity`` (older callers, tests) keeps the plain
    start-based bound."""

    monkeypatch.setattr("serviette.up._sources_look_empty", lambda _c: False)
    _fake_clock(monkeypatch)
    result = _wait_for_index(
        _config(index_wait_timeout=100.0), FakeProc([None]), ready=lambda _c, allow_empty: False
    )
    assert result is None
    assert "showed no activity for" in caplog.text
