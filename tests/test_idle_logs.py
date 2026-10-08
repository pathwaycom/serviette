"""The indexer and the server must stay quiet while nothing happens.

A colleague's first impression of an idle stack was a terminal filling with
"Persisting a chunk of 105 entries", "Done writing 0 entries" and
``GET /api/v1/stats 200 OK`` every few seconds. These tests pin the two
filters that keep those lines out while letting real activity through.
"""

from __future__ import annotations

import logging

from serviette.indexer.main import _DropIdleMonitoringLines
from serviette.server.main import _DropPollingAccessLines


def _record(name: str, msg: str, args: tuple = ()) -> logging.LogRecord:
    return logging.LogRecord(name, logging.INFO, __file__, 0, msg, args or None, None)


def test_idle_engine_heartbeats_are_dropped_but_progress_stays():
    keep = _DropIdleMonitoringLines().filter
    mon = "pathway_engine.connectors.monitoring"
    idle = [
        "DuckDB(serviette_embeddings): Done writing 0 entries, time 1791469929058. Current batch writes took: 0 ms.",
        "source_0: 0 entries (4 minibatch(es)) have been sent to the engine",
    ]
    busy = [
        "DuckDB(serviette_embeddings): Done writing 5 entries, time 1791469917058. Current batch writes took: 0 ms.",
        "source_0: 5 entries (4 minibatch(es)) have been sent to the engine",
        "source_0: 10 entries (1 minibatch(es)) have been sent to the engine",
    ]
    assert not any(keep(_record(mon, m)) for m in idle)
    assert all(keep(_record(mon, m)) for m in busy)


def test_polled_endpoints_are_kept_out_of_the_access_log():
    keep = _DropPollingAccessLines().filter
    fmt = '%s - "%s %s HTTP/%s" %d'

    def access(path, status=200):
        return _record("uvicorn.access", fmt, ("127.0.0.1:43310", "GET", path, "1.1", status))

    assert not keep(access("/api/v1/stats"))
    assert not keep(access("/api/v1/stats?x=1"))
    assert not keep(access("/health"))
    assert keep(access("/api/v1/rag"))
    assert keep(access("/api/v1/retrieve", 503))
    assert keep(access("/api/v1/documents"))
    # Records that are not uvicorn access lines pass untouched.
    assert keep(_record("uvicorn.access", "plain %s", ("x",)))
