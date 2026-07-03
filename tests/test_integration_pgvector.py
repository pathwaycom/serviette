"""Integration test: real indexer -> pgvector (Docker) -> asyncpg accessor.

Spins up a throwaway ``pgvector/pgvector`` container, runs the indexer (mock
embedder, static source) to write real vectors, then exercises the production
``PgVectorAccessor`` against it — and verifies that deleting a file on a second
indexing pass removes its rows (snapshot-mode DELETE).

Skips automatically when Docker is unavailable. Marked ``integration``/``slow``.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.dockerutil import (
    docker_available,
    free_port,
    run_container,
    stop_container,
    wait_until,
)
from vetosh.config.schema import PgVectorConfig
from vetosh.server.accessors.pgvector import PgVectorAccessor
from vetosh.testing import fake_embedding

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE = "pgvector/pgvector:pg16"
TABLE = "vetosh_embeddings"

# No schema setup here: the indexer's prepare_backend() must create the
# extension, the table and the HNSW index on its own (that is under test).


async def _can_connect(dsn: str) -> bool:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    await conn.close()
    return True


async def _count_for(dsn: str, suffix: str) -> int:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetchval(
            f"SELECT count(*) FROM {TABLE} WHERE metadata->>'path' LIKE $1",
            f"%{suffix}",
        )
    finally:
        await conn.close()


@pytest.fixture(scope="module")
def pgvector_dsn():
    if not docker_available():
        pytest.skip("Docker is not available")
    port = free_port()
    container = run_container(
        IMAGE, ports={port: 5432}, env={"POSTGRES_PASSWORD": "postgres"}
    )
    dsn = f"postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
    try:
        wait_until(lambda: asyncio.run(_can_connect(dsn)), timeout=90)
        yield dsn
    finally:
        stop_container(container)


async def _drop_table(dsn: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    finally:
        await conn.close()


@pytest.fixture(autouse=True)
def clean_table(pgvector_dsn):
    """Each test starts without the table — the indexer must create it."""

    asyncio.run(_drop_table(pgvector_dsn))
    yield


def _write_config(tmp_path: Path, docs: Path, dsn: str) -> Path:
    config = {
        "sources": [{"type": "fs", "path": str(docs), "glob": "**/*", "mode": "static"}],
        "vector_db": {"type": "pgvector", "connection_string": dsn, "table": TABLE},
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


def _run_indexer(cfg: Path, cache_dir: Path) -> None:
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "-m", "vetosh.cli", "indexer", "--config", str(cfg)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"indexer failed:\n{proc.stderr[-3000:]}"


def test_indexer_writes_and_accessor_retrieves(pgvector_dsn, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    alpha = "alpha document about cats and the streaming engine"
    (docs / "a.txt").write_text(alpha)
    (docs / "b.txt").write_text("beta report concerning dogs and the framework")
    cfg = _write_config(tmp_path, docs, pgvector_dsn)

    _run_indexer(cfg, tmp_path / "cache")

    # Both files produced rows.
    assert asyncio.run(_count_for(pgvector_dsn, "a.txt")) == 1
    assert asyncio.run(_count_for(pgvector_dsn, "b.txt")) == 1

    # The production accessor retrieves the closest chunk for a query embedding.
    async def retrieve():
        accessor = PgVectorAccessor(PgVectorConfig(connection_string=pgvector_dsn, table=TABLE))
        try:
            return await accessor.retrieve(fake_embedding(alpha), k=2)
        finally:
            await accessor.close()

    results = asyncio.run(retrieve())
    assert len(results) == 2
    assert results[0]["text"] == alpha
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert results[0]["metadata"]["path"].endswith("a.txt")


def test_deletion_removes_rows(pgvector_dsn, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha keep this document about cats")
    (docs / "b.txt").write_text("beta delete this document about dogs")
    cfg = _write_config(tmp_path, docs, pgvector_dsn)

    _run_indexer(cfg, tmp_path / "cache")
    assert asyncio.run(_count_for(pgvector_dsn, "b.txt")) == 1

    # Remove b.txt and re-run: snapshot mode must DELETE its row.
    (docs / "b.txt").unlink()
    _run_indexer(cfg, tmp_path / "cache")
    assert asyncio.run(_count_for(pgvector_dsn, "b.txt")) == 0
    assert asyncio.run(_count_for(pgvector_dsn, "a.txt")) == 1
