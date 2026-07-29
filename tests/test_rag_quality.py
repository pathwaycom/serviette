"""Tests for the retrieval-quality strategies: adaptive RAG, query
decomposition, MMR, the LLM reranker and DuckDB hybrid search."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from serviette.config.schema import (
    DuckDbConfig,
    EmbedderConfig,
    LLMConfig,
    RagConfig,
    RerankerConfig,
    ServerConfig,
    ServietteConfig,
)
from serviette.server.accessors.duckdb import DuckDbAccessor
from serviette.server.main import create_app
from serviette.server.ranking import mmr_select, rrf_merge
from serviette.server.reranker import LLMReranker, _parse_rating
from tests.conftest import fake_embedding, write_duckdb_rows

DOCS = [
    "alpha document about cats",
    "beta report on dogs",
    "gamma notes on birds",
    "delta memo on fish",
    "epsilon study of insects",
]


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


def _app(store_path, embedder, *, llm=None, rag=None, reranker=None, hybrid=False):
    config = ServietteConfig(
        vector_db=DuckDbConfig(type="duckdb", path=str(store_path), hybrid=hybrid),
        embedder=EmbedderConfig(type="openai"),
        rag=rag,
        reranker=reranker,
        server=ServerConfig(serve_frontend=False),
    )
    accessor = DuckDbAccessor(config.vector_db)
    return create_app(config, embedder=embedder, accessor=accessor, llm=llm)


# ---------------------------------------------------------------------------
# Adaptive RAG
# ---------------------------------------------------------------------------


class _AdaptiveLLM:
    """Says "no answer" until it has seen at least ``need`` context chunks."""

    def __init__(self, need: int):
        self.need = need
        self.context_sizes: list[int] = []
        self.system_prompts: list[str | None] = []

    async def complete(self, query, context, *, system_prompt=None):
        self.context_sizes.append(len(context))
        self.system_prompts.append(system_prompt)
        if len(context) < self.need:
            return "No information found in the provided context."
        return f"Real answer from {len(context)} chunks."

    async def raw(self, prompt):
        return ""

    async def close(self):
        return None


def test_adaptive_rag_grows_context_until_answered(store_path, mock_server_embedder):
    llm = _AdaptiveLLM(need=4)
    rag = RagConfig(adaptive={"factor": 2, "max_iterations": 3})
    app = _app(store_path, mock_server_embedder, llm=llm, rag=rag)
    with TestClient(app) as client:
        resp = client.post("/api/v1/rag", json={"query": "who?", "k": 1})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Real answer from 4 chunks."
    # k grew 1 -> 2 -> 4; the marker instruction was injected every round.
    assert llm.context_sizes == [1, 2, 4]
    assert all(
        p is not None and "No information found" in p for p in llm.system_prompts
    )
    assert len(resp.json()["sources"]) == 4


def test_adaptive_rag_gives_up_after_max_iterations(store_path, mock_server_embedder):
    llm = _AdaptiveLLM(need=100)  # never satisfiable on a 5-doc corpus
    rag = RagConfig(adaptive={"factor": 2, "max_iterations": 2})
    app = _app(store_path, mock_server_embedder, llm=llm, rag=rag)
    with TestClient(app) as client:
        resp = client.post("/api/v1/rag", json={"query": "who?", "k": 1})
    assert resp.status_code == 200
    assert "No information found" in resp.json()["answer"]
    assert llm.context_sizes == [1, 2]


def test_non_adaptive_rag_uses_default_prompt(store_path, mock_server_embedder):
    llm = _AdaptiveLLM(need=1)
    app = _app(store_path, mock_server_embedder, llm=llm)
    with TestClient(app) as client:
        resp = client.post("/api/v1/rag", json={"query": "who?", "k": 2})
    assert resp.status_code == 200
    assert llm.system_prompts == [None]


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------


class _DecomposingLLM(_AdaptiveLLM):
    def __init__(self, subqueries: list[str]):
        super().__init__(need=0)
        self.subqueries = subqueries
        self.raw_prompts: list[str] = []

    async def raw(self, prompt):
        self.raw_prompts.append(prompt)
        return "\n".join(self.subqueries)


def test_decompose_retrieves_for_every_subquery(store_path, mock_server_embedder):
    llm = _DecomposingLLM([DOCS[1], DOCS[2]])
    rag = RagConfig(decompose={"max_subqueries": 4})
    app = _app(store_path, mock_server_embedder, llm=llm, rag=rag)
    with TestClient(app) as client:
        resp = client.post("/api/v1/retrieve", json={"query": DOCS[0], "k": 3})
    assert resp.status_code == 200
    texts = [r["text"] for r in resp.json()["results"]]
    # Each of the three queries (original + 2 sub-queries) embeds exactly to
    # one document; fusion must surface all three despite k=3.
    assert set(texts) == {DOCS[0], DOCS[1], DOCS[2]}
    assert len(llm.raw_prompts) == 1 and DOCS[0] in llm.raw_prompts[0]


def test_decompose_requires_llm(store_path, mock_server_embedder):
    with pytest.raises(ValueError, match="rag.decompose requires"):
        _app(
            store_path,
            mock_server_embedder,
            rag=RagConfig(decompose={"max_subqueries": 3}),
        )


def test_decompose_falls_back_on_empty_llm_reply(store_path, mock_server_embedder):
    llm = _DecomposingLLM([])  # raw() returns "" -> original query only
    rag = RagConfig(decompose={"max_subqueries": 4})
    app = _app(store_path, mock_server_embedder, llm=llm, rag=rag)
    with TestClient(app) as client:
        resp = client.post("/api/v1/retrieve", json={"query": DOCS[0], "k": 1})
    assert resp.json()["results"][0]["text"] == DOCS[0]


# ---------------------------------------------------------------------------
# MMR
# ---------------------------------------------------------------------------


def test_mmr_select_prefers_novelty_over_near_duplicates():
    hits = [
        {"text": "a", "score": 1.0, "embedding": [1.0, 0.0]},
        {"text": "a-dup", "score": 0.99, "embedding": [1.0, 0.001]},
        {"text": "b", "score": 0.5, "embedding": [0.0, 1.0]},
    ]
    picked = mmr_select(hits, 2, diversity=0.5)
    assert [h["text"] for h in picked] == ["a", "b"]
    # diversity=0 must reduce to plain top-k.
    plain = mmr_select(hits, 2, diversity=0.0)
    assert [h["text"] for h in plain] == ["a", "a-dup"]


def test_mmr_endpoint_strips_embeddings(store_path, mock_server_embedder):
    rag = RagConfig(mmr={"candidates": 5, "diversity": 0.3})
    app = _app(store_path, mock_server_embedder, rag=rag)
    with TestClient(app) as client:
        resp = client.post("/api/v1/retrieve", json={"query": DOCS[0], "k": 2})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert results[0]["text"] == DOCS[0]
    assert all("embedding" not in r for r in results)


def test_mmr_rejected_for_backend_without_embeddings(store_path, mock_server_embedder):
    class NoEmbeddingsAccessor(DuckDbAccessor):
        supports_embeddings = False

    config = ServietteConfig(
        vector_db=DuckDbConfig(type="duckdb", path=str(store_path)),
        embedder=EmbedderConfig(type="openai"),
        rag=RagConfig(mmr={"candidates": 5}),
        server=ServerConfig(serve_frontend=False),
    )
    with pytest.raises(ValueError, match="rag.mmr needs hit embeddings"):
        create_app(
            config,
            embedder=mock_server_embedder,
            accessor=NoEmbeddingsAccessor(config.vector_db),
        )


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


def test_interleave_merge_guarantees_each_lists_top_hit():
    from serviette.server.ranking import interleave_merge

    list_a = [{"text": "a1", "score": 0.9}, {"text": "shared", "score": 0.8}]
    list_b = [{"text": "b1", "score": 0.2}, {"text": "shared", "score": 0.1}]
    merged = interleave_merge([list_a, list_b], 3)
    # Rank-1 of every list comes first (even with a much lower score), then
    # rank-2 deduplicated.
    assert [m["text"] for m in merged] == ["a1", "b1", "shared"]


def test_rrf_merge_rewards_presence_in_multiple_lists():
    list_a = [{"text": "x", "score": 0.9}, {"text": "y", "score": 0.8}]
    list_b = [{"text": "z", "score": 0.7}, {"text": "y", "score": 0.1}]
    merged = rrf_merge([list_a, list_b], 3)
    # "y" appears in both lists -> highest fused score despite never ranking #1.
    assert merged[0]["text"] == "y"
    assert {m["text"] for m in merged} == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# LLM reranker
# ---------------------------------------------------------------------------


class _ScoringChat:
    """Fake chat whose rating is embedded in each document's text."""

    async def raw(self, prompt: str) -> str:
        # The document line looks like "Document: rated-3 ..." -> reply "3".
        marker = prompt.rsplit("rated-", 1)
        return marker[1][0] if len(marker) == 2 else "0"

    async def close(self) -> None:
        return None


def test_llm_reranker_orders_by_llm_rating():
    import asyncio

    reranker = LLMReranker(RerankerConfig(type="llm"), chat=_ScoringChat())
    hits = [
        {"text": "rated-1 low", "score": 0.9},
        {"text": "rated-5 high", "score": 0.1},
        {"text": "rated-3 mid", "score": 0.5},
    ]
    reranked = asyncio.run(reranker.rerank("q", hits, 2))
    assert [h["text"] for h in reranked] == ["rated-5 high", "rated-3 mid"]
    assert [h["score"] for h in reranked] == [5.0, 3.0]


def test_parse_rating():
    assert _parse_rating("4") == 4.0
    assert _parse_rating("Score: 3.5/5") == 3.5
    assert _parse_rating("no idea") == 0.0


def test_llm_reranker_via_config_builds(store_path, mock_server_embedder, mock_llm):
    """create_app accepts reranker type 'llm' (chat built lazily, no API call)."""

    reranker = RerankerConfig(type="llm", candidates=3)
    config = ServietteConfig(
        vector_db=DuckDbConfig(type="duckdb", path=str(store_path)),
        embedder=EmbedderConfig(type="openai"),
        llm=LLMConfig(type="openai", api_key="test-key"),
        reranker=reranker,
        server=ServerConfig(serve_frontend=False),
    )
    accessor = DuckDbAccessor(config.vector_db)
    app = create_app(
        config, embedder=mock_server_embedder, accessor=accessor, llm=mock_llm
    )
    assert app is not None


# ---------------------------------------------------------------------------
# DuckDB hybrid search
# ---------------------------------------------------------------------------


def test_hybrid_finds_keyword_match_vector_search_misses(tmp_path, mock_server_embedder):
    # The fake embedding is a content hash: unrelated texts are uncorrelated,
    # so a query that shares *words* (but not the exact text) with a document
    # ranks arbitrarily in vector order — BM25 must surface it.
    path = tmp_path / "store.duckdb"
    docs = DOCS + ["the treaty of Guadalupe Hidalgo was signed in 1848"]
    write_duckdb_rows(
        path,
        [
            {
                "id": str(i),
                "text": text,
                "metadata": {"path": f"/docs/{i}.txt"},
                "embedding": fake_embedding(text),
            }
            for i, text in enumerate(docs)
        ],
    )
    app = _app(path, mock_server_embedder, hybrid=True)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/retrieve",
            json={"query": "when was Guadalupe Hidalgo signed", "k": 3},
        )
    assert resp.status_code == 200
    texts = [r["text"] for r in resp.json()["results"]]
    assert any("Guadalupe Hidalgo" in t for t in texts)


def test_hybrid_off_keeps_pure_vector_behavior(store_path, mock_server_embedder):
    app = _app(store_path, mock_server_embedder, hybrid=False)
    with TestClient(app) as client:
        resp = client.post("/api/v1/retrieve", json={"query": DOCS[0], "k": 1})
    results = resp.json()["results"]
    assert results[0]["text"] == DOCS[0]
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)


def test_hybrid_index_rebuilds_when_rows_change(tmp_path, mock_server_embedder):
    path = tmp_path / "store.duckdb"
    rows = [
        {
            "id": str(i),
            "text": text,
            "metadata": {},
            "embedding": fake_embedding(text),
        }
        for i, text in enumerate(DOCS)
    ]
    write_duckdb_rows(path, rows)
    app = _app(path, mock_server_embedder, hybrid=True)
    with TestClient(app) as client:
        first = client.post(
            "/api/v1/retrieve", json={"query": "zanzibar spice market", "k": 2}
        )
        texts = [r["text"] for r in first.json()["results"]]
        assert not any("zanzibar" in t for t in texts)
        write_duckdb_rows(
            path,
            [
                {
                    "id": "new",
                    "text": "zanzibar spice market report",
                    "metadata": {},
                    "embedding": fake_embedding("zanzibar spice market report"),
                }
            ],
        )
        second = client.post(
            "/api/v1/retrieve", json={"query": "zanzibar spice market", "k": 2}
        )
        texts = [r["text"] for r in second.json()["results"]]
        assert any("zanzibar" in t for t in texts)
