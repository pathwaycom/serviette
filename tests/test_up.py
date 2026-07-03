"""End-to-end test for ``vetosh up`` (supervisor semantics).

Uses static sources + DuckDB + the mock embedder: the indexer child finishes
its one-shot pass and exits 0, the server child keeps serving — exercising
spawn, the "indexer done, server continues" branch, HTTP serving and clean
SIGINT teardown of the whole process group.
"""

from __future__ import annotations

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
        try:
            if predicate():
                return
        except Exception:  # noqa: BLE001 - readiness probing
            pass
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
        [sys.executable, "-m", "vetosh.cli", "up", "--config", str(cfg)],
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
