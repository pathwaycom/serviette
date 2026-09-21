"""Unit tests for the server-readiness wait in ``serviette up`` (no Pathway).

``up`` must announce the URL only once the server answers its health check:
uvicorn binds the port after the lifespan warm-up, and a local embedder's
warm-up (torch import + model load) keeps the port closed for tens of
seconds — a URL printed at spawn time sends the user to "connection refused".
"""

from __future__ import annotations

import pytest

from serviette.config.schema import ServietteConfig
from serviette.up import _server_url, _wait_for_server


class FakeProc:
    def __init__(self, codes):
        # Successive poll() results; the last one repeats.
        self._codes = list(codes)

    def poll(self):
        if len(self._codes) > 1:
            return self._codes.pop(0)
        return self._codes[0]


def _config(host: str = "127.0.0.1", port: int = 8989) -> ServietteConfig:
    return ServietteConfig.model_validate(
        {
            "sources": [{"type": "fs", "path": "."}],
            "vector_db": {"type": "duckdb", "path": "e.duckdb"},
            "embedder": {"type": "mock"},
            "server": {"host": host, "port": port},
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
