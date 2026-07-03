"""Shared scenario driver for the per-backend integration tests.

Every backend runs the same story through **production code paths**:

1. index two documents (real indexer subprocess, mock embedder, static mode);
2. retrieve through the backend's production accessor — the exact-text query
   must rank its document first with a cosine score of ~1.0;
3. delete one file, re-run the indexer, and verify its vectors are gone from
   the store (snapshot semantics), again through the accessor.

Backends differ only in their ``vector_db`` config section and any target
pre-creation (collection/index), which each test file provides.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from vetosh.server.accessors.abstract import AsyncVectorAccessor
from vetosh.testing import fake_embedding

REPO_ROOT = Path(__file__).resolve().parent.parent

ALPHA = "alpha document about cats and the streaming engine"
BETA = "beta report concerning dogs and the framework"


def write_config(tmp_path: Path, docs: Path, vector_db: dict[str, Any]) -> Path:
    config = {
        "sources": [{"type": "fs", "path": str(docs), "glob": "**/*", "mode": "static"}],
        "vector_db": vector_db,
        "embedder": {"type": "mock"},
        "splitter": {"type": "token_count", "chunk_size": 512, "chunk_overlap": 50},
        "persistence": {
            "enabled": True,
            "backend": "filesystem",
            "path": str(tmp_path / "persist"),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def run_indexer(cfg: Path, _legacy_cache_dir: Path | None = None) -> None:
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "-m", "vetosh.cli", "indexer", "--config", str(cfg)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, f"indexer failed:\n{proc.stderr[-3000:]}"


def _retrieve(make_accessor: Callable[[], AsyncVectorAccessor], text: str, k: int):
    async def go():
        accessor = make_accessor()
        try:
            return await accessor.retrieve(fake_embedding(text), k)
        finally:
            await accessor.close()

    return asyncio.run(go())


def paths_of(results: list[dict[str, Any]]) -> set[str]:
    return {Path(r["metadata"]["path"]).name for r in results if r.get("metadata")}


def run_backend_scenario(
    tmp_path: Path,
    vector_db: dict[str, Any],
    make_accessor: Callable[[], AsyncVectorAccessor],
    *,
    after_index: Callable[[], None] | None = None,
    score_abs: float = 1e-4,
) -> None:
    """Run the full index → retrieve → delete → verify scenario.

    ``after_index`` runs after each indexing pass — for backends that need a
    post-index step before querying (e.g. MongoDB search-index readiness).
    """

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(ALPHA)
    (docs / "b.txt").write_text(BETA)
    cfg = write_config(tmp_path, docs, vector_db)

    # -- pass 1: both documents land and are retrievable ----------------------
    run_indexer(cfg, tmp_path / "cache")
    if after_index is not None:
        after_index()

    results = _retrieve(make_accessor, ALPHA, k=2)
    assert len(results) == 2
    assert results[0]["text"] == ALPHA
    assert results[0]["score"] == pytest.approx(1.0, abs=score_abs)
    assert results[0]["metadata"]["path"].endswith("a.txt")
    assert paths_of(results) == {"a.txt", "b.txt"}

    # -- pass 2: deleting b.txt removes its vectors (snapshot semantics) ------
    (docs / "b.txt").unlink()
    run_indexer(cfg, tmp_path / "cache")
    if after_index is not None:
        after_index()

    results = _retrieve(make_accessor, BETA, k=5)
    assert paths_of(results) == {"a.txt"}, "b.txt vectors must be deleted"
