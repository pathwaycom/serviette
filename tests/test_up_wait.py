"""Unit tests for the server-readiness wait in ``serviette up`` (no Pathway).

``up`` must announce the URL only once the server answers its health check:
uvicorn binds the port after the lifespan warm-up, and a local embedder's
warm-up (torch import + model load) keeps the port closed for tens of
seconds — a URL printed at spawn time sends the user to "connection refused".
"""

from __future__ import annotations

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

    def fake_spawn(command, config_path, *, env=None):
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
