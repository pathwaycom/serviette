"""Tests for the FastAPI server (uses the DuckDB accessor + mock embedder)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import fake_embedding, write_duckdb_rows
from vetosh.config.schema import DuckDbConfig, EmbedderConfig, VetoshConfig
from vetosh.server.accessors.duckdb import DuckDbAccessor
from vetosh.server.main import create_app

DOCS = ["alpha document about cats", "beta report on dogs", "gamma notes on birds"]


@pytest.fixture
def store_path(tmp_path):
    path = tmp_path / "store.duckdb"
    rows = [
        {
            "id": str(i),
            "text": text,
            "metadata": {"path": f"/docs/{i}.txt", "idx": i},
            "embedding": fake_embedding(text),
        }
        for i, text in enumerate(DOCS)
    ]
    write_duckdb_rows(path, rows)
    return path


def _client(store_path, embedder, llm=None):
    config = VetoshConfig(
        vector_db=DuckDbConfig(type="duckdb", path=str(store_path)),
        embedder=EmbedderConfig(type="openai"),
    )
    accessor = DuckDbAccessor(config.vector_db)
    app = create_app(config, embedder=embedder, accessor=accessor, llm=llm)
    return TestClient(app)


def test_retrieve_top_k(store_path, mock_server_embedder):
    with _client(store_path, mock_server_embedder) as client:
        resp = client.post("/api/v1/retrieve", json={"query": DOCS[0], "k": 2})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    # Exact-text query embeds to the same vector -> top hit is that document.
    assert results[0]["text"] == DOCS[0]
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)
    assert results[0]["metadata"]["idx"] == 0


def test_results_ordered_by_score_desc(store_path, mock_server_embedder):
    with _client(store_path, mock_server_embedder) as client:
        resp = client.post("/api/v1/retrieve", json={"query": DOCS[1], "k": 3})
    scores = [r["score"] for r in resp.json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_rag_returns_answer(store_path, mock_server_embedder, mock_llm):
    with _client(store_path, mock_server_embedder, llm=mock_llm) as client:
        resp = client.post("/api/v1/rag", json={"query": "tell me about cats", "k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["answer"], str) and body["answer"]
    assert len(body["sources"]) == 2


def test_rag_disabled_without_llm(store_path, mock_server_embedder):
    with _client(store_path, mock_server_embedder) as client:
        resp = client.post("/api/v1/rag", json={"query": "x", "k": 1})
    assert resp.status_code == 501


def test_legacy_unversioned_aliases_still_work(store_path, mock_server_embedder):
    """Pre-versioning routes are kept as deprecated aliases of /api/v1."""

    with _client(store_path, mock_server_embedder) as client:
        health = client.get("/health")
        legacy = client.post("/retrieve", json={"query": DOCS[0], "k": 1})
        v1 = client.post("/api/v1/retrieve", json={"query": DOCS[0], "k": 1})
    assert health.status_code == 200
    assert legacy.status_code == 200
    assert legacy.json() == v1.json()


def test_serves_embedded_chat_page(store_path, mock_server_embedder):
    with _client(store_path, mock_server_embedder) as client:
        resp = client.get("/")
        cfg = client.get("/api/v1/config").json()
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "__APP_TITLE__" not in resp.text
    # Empty api_url means "same origin" for the page.
    assert cfg == {"title": "vetosh", "api_url": ""}


def test_stats_endpoint(store_path, mock_server_embedder):
    with _client(store_path, mock_server_embedder) as client:
        body = client.get("/api/v1/stats").json()
    # Three rows over three distinct source paths (see the store fixture).
    assert body["backend"] == "duckdb"
    assert body["chunks"] == 3
    assert body["documents"] == 3


def test_frontend_can_be_disabled(store_path, mock_server_embedder):
    config = VetoshConfig(
        vector_db=DuckDbConfig(type="duckdb", path=str(store_path)),
        embedder=EmbedderConfig(type="openai"),
        server={"serve_frontend": False},
    )
    accessor = DuckDbAccessor(config.vector_db)
    app = create_app(config, embedder=mock_server_embedder, accessor=accessor)
    with TestClient(app) as client:
        assert client.get("/").status_code == 404
        assert client.get("/api/v1/health").status_code == 200


def test_index_not_ready_returns_friendly_503(tmp_path, mock_server_embedder):
    """Standalone server before the indexer ran: 503 + a human message.

    (The orchestrated path never hits this: `vetosh up` holds the server
    back until the store is queryable.)
    """
    missing = tmp_path / "never-created.duckdb"
    with _client(missing, mock_server_embedder) as client:
        resp = client.post("/api/v1/retrieve", json={"query": "hello", "k": 2})
    assert resp.status_code == 503
    body = resp.json()
    assert "not ready" in body["detail"]
    assert resp.headers.get("retry-after") == "5"


def test_local_embedder_warmed_up_at_startup(store_path):
    """A local embedder is exercised once by the lifespan, before any request."""

    class CountingLocalEmbedder:
        is_local = True
        calls = 0

        async def embed(self, text):
            type(self).calls += 1
            return fake_embedding(text)

        async def close(self):
            return None

    embedder = CountingLocalEmbedder()
    with _client(store_path, embedder):
        assert CountingLocalEmbedder.calls == 1  # warm-up ran on startup


def test_cors_disabled_by_default(store_path, mock_server_embedder):
    with _client(store_path, mock_server_embedder) as client:
        resp = client.post(
            "/api/v1/retrieve",
            json={"query": "hello", "k": 1},
            headers={"Origin": "https://evil.example"},
        )
    assert "access-control-allow-origin" not in resp.headers


def test_cors_opt_in_allowlist(store_path, mock_server_embedder):
    config = VetoshConfig(
        vector_db=DuckDbConfig(type="duckdb", path=str(store_path)),
        embedder=EmbedderConfig(type="openai"),
        server={"cors_origins": ["https://app.example.com"]},
    )
    accessor = DuckDbAccessor(config.vector_db)
    app = create_app(config, embedder=mock_server_embedder, accessor=accessor)
    with TestClient(app) as client:
        allowed = client.post(
            "/api/v1/retrieve",
            json={"query": "hello", "k": 1},
            headers={"Origin": "https://app.example.com"},
        )
        assert allowed.headers.get("access-control-allow-origin") == "https://app.example.com"
        foreign = client.post(
            "/api/v1/retrieve",
            json={"query": "hello", "k": 1},
            headers={"Origin": "https://evil.example"},
        )
        assert "access-control-allow-origin" not in foreign.headers
