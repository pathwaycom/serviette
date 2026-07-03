"""Integration test: real indexer -> Milvus -> async accessor.

Uses **Milvus Lite** (the embedded Milvus engine in the ``milvus-lite`` package)
so it exercises the real ``pw.io.milvus.write`` sink and the production
``MilvusAccessor`` (``AsyncMilvusClient``) without orchestrating a multi-service
Milvus standalone Docker stack. The same code paths run against a real Milvus
server — only the ``uri`` differs (``http://host:19530``).

Every Milvus operation runs in its own short-lived subprocess (see
``tests/_milvus_writer.py``): Milvus Lite releases its file lock only on process
exit, so sharing the file across a long-lived test process is unreliable.

Milvus Lite does not honour the COSINE metric (it ranks by L2), so we assert on
relevance *ordering*/identity, not on the absolute similarity score; against a
real COSINE-indexed Milvus server the accessor returns a true cosine similarity.

Marked ``integration``/``slow``; skips if ``milvus-lite`` is not installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("milvus_lite", reason="milvus-lite not installed")
pytest.importorskip("pymilvus", reason="pymilvus not installed")

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parent.parent
COLLECTION = "vetosh_embeddings"


def _config(tmp_path: Path, docs: Path, uri: str) -> dict:
    return {
        "sources": [{"type": "fs", "path": str(docs), "glob": "**/*", "mode": "static"}],
        "vector_db": {"type": "milvus", "uri": uri, "collection": COLLECTION},
        "embedder": {"type": "mock"},
        "splitter": {"type": "token_count", "chunk_size": 512, "chunk_overlap": 50},
        "persistence": {
            "enabled": True,
            "backend": "filesystem",
            "path": str(tmp_path / "persist"),
        },
    }


def _driver(args: list[str], cache_dir: Path, attempts: int = 4) -> object:
    """Invoke the Milvus driver subprocess, retrying the known Lite startup flake."""

    env = dict(os.environ)
    last = ""
    for _ in range(attempts):
        proc = subprocess.run(
            [sys.executable, "-m", "tests._milvus_writer", *args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode == 0:
            marker = "__RESULT__"
            line = next(ln for ln in proc.stdout.splitlines() if ln.startswith(marker))
            return json.loads(line[len(marker):])
        last = proc.stderr[-2000:]
        flaky = "failed to start" in last or "DataDirLocked" in last
        if not flaky:
            break
    raise AssertionError(f"milvus driver {args[0]!r} failed:\n{last}")


def _write(uri, config, cache_dir):
    _driver(["write", uri, COLLECTION, json.dumps(config)], cache_dir)


def _paths(uri, cache_dir) -> list[str]:
    return _driver(["paths", uri, COLLECTION], cache_dir)


def _retrieve(uri, query, cache_dir) -> list[dict]:
    return _driver(["retrieve", uri, COLLECTION, query], cache_dir)


@pytest.fixture
def uri(tmp_path):
    return str(tmp_path / "milvus.db")


def test_write_and_retrieve_ranking(uri, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    alpha = "alpha document about cats and the streaming engine"
    (docs / "a.txt").write_text(alpha)
    (docs / "b.txt").write_text("beta report concerning dogs and the framework")
    cache = tmp_path / "cache"

    _write(uri, _config(tmp_path, docs, uri), cache)
    assert _paths(uri, cache) == ["a.txt", "b.txt"]

    results = _retrieve(uri, alpha, cache)
    assert len(results) == 2
    # The exact-text query ranks its own document first.
    assert results[0]["metadata"]["path"].endswith("a.txt")
    assert results[0]["text"] == alpha


def test_deletion_removes_rows(uri, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha keep this document about cats")
    (docs / "b.txt").write_text("beta delete this document about dogs")
    config = _config(tmp_path, docs, uri)
    cache = tmp_path / "cache"

    _write(uri, config, cache)
    assert _paths(uri, cache) == ["a.txt", "b.txt"]

    (docs / "b.txt").unlink()
    _write(uri, config, cache)
    assert _paths(uri, cache) == ["a.txt"]  # b's vectors removed
