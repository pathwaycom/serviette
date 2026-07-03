"""End-to-end indexer tests.

Each indexing pass runs in a fresh subprocess (Pathway builds a single global
graph per process, and a restart is exactly how persistence-driven deletion is
exercised). Passes use ``mode: static`` so the graph terminates, the test-only
``mock`` embedder (no credentials), and the embedded DuckDB sink/accessor.

These are slower than the unit tests (~30-40s of Pathway startup per pass); run
with ``-m "not slow"`` to skip.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parent.parent
PASS_TIMEOUT = 180


def _write_config(tmp_path, docs_dir, store, persist_dir, *, persistence: bool) -> Path:
    config = {
        "sources": [
            {"type": "fs", "path": str(docs_dir), "glob": "**/*", "mode": "static"}
        ],
        "vector_db": {"type": "duckdb", "path": str(store), "table": "vetosh_embeddings"},
        "embedder": {"type": "mock"},
        "splitter": {"type": "token_count", "chunk_size": 512, "chunk_overlap": 50},
        "persistence": {
            "enabled": persistence,
            "backend": "filesystem",
            "path": str(persist_dir),
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    return cfg_path


def _run_pass(cfg_path: Path, cache_dir: Path | None = None) -> None:
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "-m", "vetosh.cli", "indexer", "--config", str(cfg_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=PASS_TIMEOUT,
    )
    assert proc.returncode == 0, f"indexer failed:\n{proc.stderr[-3000:]}"


def _read_store(store: Path) -> list[dict]:
    if not store.exists():
        return []
    import duckdb

    conn = duckdb.connect(str(store), read_only=True)
    try:
        rows = conn.execute(
            "SELECT chunk_id, text, metadata FROM vetosh_embeddings"
        ).fetchall()
    finally:
        conn.close()
    return [
        {"id": r[0], "text": r[1], "metadata": json.loads(r[2])} for r in rows
    ]


def _files_in_store(store: Path) -> set[str]:
    return {Path(r["metadata"]["path"]).name for r in _read_store(store)}


@pytest.fixture
def env(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    return {
        "tmp": tmp_path,
        "docs": docs,
        "store": tmp_path / "store.duckdb",
        "persist": tmp_path / "persist",
        "cache": tmp_path / "cache",
    }


def test_add_modify_delete(env):
    docs, store = env["docs"], env["store"]
    cfg = _write_config(
        env["tmp"], docs, store, env["persist"], persistence=True
    )

    # Pass 1: index two files -> vectors for both appear.
    (docs / "a.txt").write_text("alpha document about cats and the streaming engine")
    (docs / "b.txt").write_text("beta report concerning dogs and live data framework")
    _run_pass(cfg, env["cache"])
    assert _files_in_store(store) == {"a.txt", "b.txt"}
    a_ids_before = {r["id"] for r in _read_store(store) if r["metadata"]["path"].endswith("a.txt")}
    assert a_ids_before

    # Pass 2: delete b (vectors removed), add c (new vectors), modify a
    # (old vectors removed, new ones added).
    (docs / "b.txt").unlink()
    (docs / "c.txt").write_text("gamma notes on birds and incremental computation")
    (docs / "a.txt").write_text("alpha REWRITTEN entirely new content about felines now")
    _run_pass(cfg, env["cache"])

    assert _files_in_store(store) == {"a.txt", "c.txt"}  # b deleted, c added
    a_ids_after = {r["id"] for r in _read_store(store) if r["metadata"]["path"].endswith("a.txt")}
    assert a_ids_after and a_ids_after.isdisjoint(a_ids_before)  # a's vectors replaced


def test_persistence_on_and_off_give_same_vectors(tmp_path):
    """For a fixed file set, the resulting vectors are identical regardless of
    whether Pathway persistence is enabled."""

    def index(persistence: bool) -> set[str]:
        root = tmp_path / ("on" if persistence else "off")
        docs = root / "docs"
        docs.mkdir(parents=True)
        (docs / "a.txt").write_text("alpha document about cats")
        (docs / "b.txt").write_text("beta report about dogs and frameworks")
        store = root / "store.duckdb"
        cfg = _write_config(root, docs, store, root / "persist", persistence=persistence)
        _run_pass(cfg, root / "cache")
        return {(r["text"], r["metadata"]["path"].split("/")[-1]) for r in _read_store(store)}

    with_persist = index(True)
    without_persist = index(False)
    assert with_persist == without_persist and with_persist
