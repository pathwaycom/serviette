"""Shared test fixtures.

Kept import-light at module scope: ``pathway`` is imported lazily inside the
fixtures that need it, so the cache/server/wizard tests run without pulling
in the heavy Pathway/xpack stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Single source of truth for the deterministic fake embedding, shared with the
# indexer's test-only "mock" embedder so indexer and server vectors agree.
from serviette.testing import EMBED_DIM, fake_embedding  # noqa: F401


class MockServerEmbedder:
    """Mock implementing the server's :class:`AsyncEmbedder` protocol."""

    async def embed(self, text: str) -> list[float]:
        return fake_embedding(text)

    async def close(self) -> None:
        return None


class MockLLM:
    """Mock implementing the server's :class:`AsyncLLM` protocol."""

    async def complete(
        self, query: str, context: list[str], *, system_prompt: str | None = None
    ) -> str:
        return f"Answer to {query!r} from {len(context)} source(s)."

    async def raw(self, prompt: str) -> str:
        return ""

    async def close(self) -> None:
        return None


@pytest.fixture
def tcp_port() -> int:
    """A dynamically allocated free TCP port (never hardcode host ports)."""

    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def tcp_port_factory():
    """Factory flavour of ``tcp_port`` for tests that need several ports
    (e.g. a service exposing both REST and gRPC)."""

    import socket

    def make() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    return make


@pytest.fixture
def mock_server_embedder() -> MockServerEmbedder:
    return MockServerEmbedder()


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def mock_xpack_embedder():
    """A Pathway UDF mock embedder for the indexer graph (returns ndarray)."""

    from serviette.testing import build_mock_embedder

    return build_mock_embedder()


def write_duckdb_rows(path: Path, rows: list[dict]) -> None:
    """Populate a DuckDB vector store directly (server tests).

    Mirrors the schema the indexer's ``pw.io.duckdb`` snapshot sink creates:
    ``(chunk_id, text, metadata JSON-string, embedding DOUBLE[])``.
    """

    import duckdb

    conn = duckdb.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS serviette_embeddings ("
        "  chunk_id VARCHAR PRIMARY KEY,"
        "  text VARCHAR,"
        "  metadata VARCHAR,"
        "  embedding DOUBLE[]"
        ")"
    )
    for row in rows:
        conn.execute(
            "INSERT OR REPLACE INTO serviette_embeddings VALUES (?, ?, ?, ?)",
            (
                row["id"],
                row["text"],
                json.dumps(row.get("metadata", {})),
                [float(x) for x in row["embedding"]],
            ),
        )
    conn.close()
