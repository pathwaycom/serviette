"""End-to-end test for ``serviette up`` (supervisor semantics).

Uses static sources + DuckDB + the mock embedder: the indexer child finishes
its one-shot pass and exits 0, the server child keeps serving — exercising
spawn, the "indexer done, server continues" branch, HTTP serving and clean
SIGINT teardown of the whole process group.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parent.parent


def _wait(predicate, timeout: float, message: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.suppress(Exception):
            if predicate():
                return
        time.sleep(0.5)
    raise TimeoutError(message)


def test_up_serves_after_static_indexing(tmp_path, tcp_port):
    docs = tmp_path / "docs"
    docs.mkdir()
    alpha = "alpha document about cats and the streaming engine"
    (docs / "a.txt").write_text(alpha)

    config = {
        "sources": [{"type": "fs", "path": str(docs), "glob": "**/*", "mode": "static"}],
        "vector_db": {"type": "duckdb", "path": str(tmp_path / "store.duckdb")},
        "embedder": {"type": "mock"},
        "persistence": {"enabled": False},
        "server": {"host": "127.0.0.1", "port": tcp_port},
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))

    env = dict(os.environ)
    proc = subprocess.Popen(
        [sys.executable, "-m", "serviette.cli", "up", "--config", str(cfg)],
        cwd=REPO_ROOT,
        env=env,
    )
    base = f"http://127.0.0.1:{tcp_port}"
    try:
        _wait(
            lambda: httpx.get(f"{base}/api/v1/health").status_code == 200,
            timeout=120,
            message="server did not come up",
        )
        # The chat UI is served from the same port.
        assert "<html" in httpx.get(base + "/").text.lower()

        # Retrieval works once the (static) indexing pass lands; the accessor
        # retries through the brief indexer-holds-the-file window.
        def indexed() -> bool:
            resp = httpx.post(
                f"{base}/api/v1/retrieve",
                json={"query": alpha, "k": 1},
                timeout=30,
            )
            return resp.status_code == 200 and bool(resp.json()["results"])

        _wait(indexed, timeout=120, message="documents were not indexed")

        # Clean teardown on SIGINT; no orphaned children.
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_index_ready_probe_duckdb(tmp_path):
    """_index_ready: False before the indexer created anything, True after."""
    import duckdb

    from serviette.config.schema import ServietteConfig
    from serviette.up import _index_ready

    config = ServietteConfig.model_validate(
        {
            "sources": [{"type": "fs", "path": str(tmp_path)}],
            "vector_db": {
                "type": "duckdb",
                "path": str(tmp_path / "e.duckdb"),
                "table": "embeddings",
            },
            "embedder": {"type": "mock"},
        }
    )
    assert not _index_ready(config)  # file does not exist yet

    conn = duckdb.connect(str(tmp_path / "e.duckdb"))
    assert not _index_ready(config)  # file exists, table does not

    conn.execute(
        "CREATE TABLE embeddings (chunk_id VARCHAR, text VARCHAR, "
        "metadata VARCHAR, embedding DOUBLE[])"
    )
    assert not _index_ready(config)  # prepared but still empty: no chunks yet

    conn.execute(
        "INSERT INTO embeddings VALUES ('c1', 'hello', "
        "'{\"path\": \"/a.txt\"}', [0.1, 0.2])"
    )
    conn.close()
    assert _index_ready(config)  # first real content -> server may start


def test_sources_look_empty(tmp_path):
    from serviette.config.schema import ServietteConfig
    from serviette.up import _sources_look_empty

    def cfg(path):
        return ServietteConfig.model_validate(
            {
                "sources": [{"type": "fs", "path": str(path)}],
                "vector_db": {"type": "duckdb", "path": str(tmp_path / "e.duckdb")},
                "embedder": {"type": "mock"},
            }
        )

    docs = tmp_path / "docs"
    docs.mkdir()
    assert _sources_look_empty(cfg(docs))  # empty folder
    (docs / "sub").mkdir()
    assert _sources_look_empty(cfg(docs))  # directories don't count
    (docs / "sub" / "a.txt").write_text("hi")
    assert not _sources_look_empty(cfg(docs))  # a real file

    remote = ServietteConfig.model_validate(
        {
            "sources": [
                {"type": "fs", "path": str(docs)},
                {"type": "s3", "bucket": "b"},
            ],
            "vector_db": {"type": "duckdb", "path": str(tmp_path / "e.duckdb")},
            "embedder": {"type": "mock"},
        }
    )
    assert not _sources_look_empty(remote)  # remote sources: assume non-empty


def test_prepare_duckdb_creates_empty_queryable_store(tmp_path):
    from serviette.config.schema import ServietteConfig
    from serviette.indexer.prepare import prepare_backend
    from serviette.up import _index_ready

    config = ServietteConfig.model_validate(
        {
            "sources": [{"type": "fs", "path": str(tmp_path)}],
            "vector_db": {
                "type": "duckdb",
                "path": str(tmp_path / "store" / "e.duckdb"),
                "table": "embeddings",
            },
            "embedder": {"type": "mock"},
        }
    )
    prepare_backend(config)
    # Queryable immediately (allow_empty), and idempotent.
    prepare_backend(config)
    assert _index_ready(config, allow_empty=True)
    assert not _index_ready(config)  # still no chunks


def test_up_empty_folder_starts_and_indexes_live(tmp_path, tcp_port):
    """An empty source folder must not block `serviette up`: the server starts
    over the pre-created empty DuckDB store, /retrieve answers (empty), and a
    file dropped in afterwards is indexed live — proving the writer accepts
    the pre-created table (schema parity) end-to-end."""
    import signal as _signal
    import subprocess
    import sys
    import time

    import httpx
    import yaml

    docs = tmp_path / "docs"
    docs.mkdir()
    config = {
        "sources": [{"type": "fs", "path": str(docs), "glob": "**/*"}],
        "vector_db": {
            "type": "duckdb",
            "path": str(tmp_path / "e.duckdb"),
            "table": "embeddings",
        },
        "embedder": {"type": "mock"},
        "server": {"host": "127.0.0.1", "port": tcp_port},
        "persistence": {"enabled": False},
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    proc = subprocess.Popen(
        [sys.executable, "-m", "serviette.cli", "up", "--config", str(cfg)],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONPATH=f"{REPO_ROOT}{os.pathsep}" + os.environ.get("PYTHONPATH", ""),
        ),
    )
    base = f"http://127.0.0.1:{tcp_port}"
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base}/api/v1/health", timeout=5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            assert proc.poll() is None, "up exited prematurely"
            time.sleep(1)
        else:
            raise TimeoutError("server did not start over the empty index")

        resp = httpx.post(f"{base}/api/v1/retrieve", json={"query": "x", "k": 3}, timeout=30)
        assert resp.status_code == 200 and resp.json()["results"] == []

        (docs / "late.txt").write_text("dogs bark at the postman every morning")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            resp = httpx.post(
                f"{base}/api/v1/retrieve", json={"query": "dogs bark", "k": 3}, timeout=30
            )
            if resp.status_code == 200 and resp.json()["results"]:
                break
            time.sleep(2)
        else:
            raise TimeoutError("late-added file was not indexed")

        proc.send_signal(_signal.SIGINT)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
