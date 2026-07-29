"""Unit tests for the Qdrant accessor's hybrid (BM25) mode and embedding
support, using a stub client — the real-server path is covered by the
integration suite."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from serviette.config.schema import QdrantConfig
from serviette.server.accessors.qdrant import QdrantAccessor
from tests.conftest import fake_embedding

DOCS = [
    "alpha document about cats",
    "beta report on dogs",
    "gamma notes on birds",
    "the treaty of Guadalupe Hidalgo was signed in 1848",
]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)


class StubQdrantClient:
    """Mimics the AsyncQdrantClient subset the accessor uses."""

    def __init__(self, docs: list[str]):
        self.points = [
            SimpleNamespace(
                id=i,
                payload={"text": text, "metadata": {"idx": i}},
                vector=fake_embedding(text),
                score=None,
            )
            for i, text in enumerate(docs)
        ]
        self.scroll_calls = 0

    async def query_points(
        self, collection_name, query, limit, with_payload, with_vectors=False, using=None
    ):
        scored = sorted(
            self.points, key=lambda p: -_cosine(p.vector, query)
        )[:limit]
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    payload=p.payload,
                    vector=p.vector if with_vectors else None,
                    score=_cosine(p.vector, query),
                )
                for p in scored
            ]
        )

    async def count(self, collection_name, exact):
        return SimpleNamespace(count=len(self.points))

    async def scroll(
        self, collection_name, limit, offset, with_payload, with_vectors=False
    ):
        self.scroll_calls += 1
        start = offset or 0
        page = self.points[start : start + limit]
        next_offset = start + limit if start + limit < len(self.points) else None
        return (
            [
                SimpleNamespace(
                    payload=p.payload,
                    vector=p.vector if with_vectors else None,
                    score=None,
                )
                for p in page
            ],
            next_offset,
        )

    async def close(self):
        return None


def _accessor(docs=DOCS, **config_kwargs) -> tuple[QdrantAccessor, StubQdrantClient]:
    config = QdrantConfig(type="qdrant", **config_kwargs)
    accessor = QdrantAccessor(config)
    stub = StubQdrantClient(docs)
    accessor._client = stub
    return accessor, stub


def test_hybrid_surfaces_keyword_match_vector_search_misses():
    accessor, _ = _accessor(hybrid=True)
    hits = asyncio.run(
        accessor.retrieve_ex(
            fake_embedding("some unrelated query"),
            3,
            query_text="when was Guadalupe Hidalgo signed",
        )
    )
    assert any("Guadalupe Hidalgo" in h["text"] for h in hits)


def test_hybrid_off_is_pure_vector():
    accessor, stub = _accessor(hybrid=False)
    hits = asyncio.run(
        accessor.retrieve_ex(
            fake_embedding(DOCS[0]), 2, query_text="whatever keywords"
        )
    )
    assert hits[0]["text"] == DOCS[0]
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-6)
    assert stub.scroll_calls == 0


def test_hybrid_reuses_index_until_count_changes():
    accessor, stub = _accessor(hybrid=True)

    async def run():
        await accessor.retrieve_ex(fake_embedding("q1"), 2, query_text="cats")
        await accessor.retrieve_ex(fake_embedding("q2"), 2, query_text="dogs")
        calls_before_change = stub.scroll_calls
        stub.points.append(
            SimpleNamespace(
                id=len(stub.points),
                payload={"text": "zanzibar spice market", "metadata": {}},
                vector=fake_embedding("zanzibar spice market"),
                score=None,
            )
        )
        hits = await accessor.retrieve_ex(
            fake_embedding("q3"), 2, query_text="zanzibar market"
        )
        return calls_before_change, hits

    calls_before_change, hits = asyncio.run(run())
    # 4 docs / page 1024 -> one scroll for the first build, none for the
    # second query, one more after the count changed.
    assert calls_before_change == 1
    assert stub.scroll_calls == 2
    assert any("zanzibar" in h["text"] for h in hits)


def test_hybrid_skips_bm25_above_max_chunks(caplog):
    accessor, stub = _accessor(hybrid=True, hybrid_max_chunks=2)
    hits = asyncio.run(
        accessor.retrieve_ex(
            fake_embedding(DOCS[0]), 2, query_text="Guadalupe Hidalgo"
        )
    )
    # Falls back to pure vector search: no scroll, warning logged once.
    assert stub.scroll_calls == 0
    assert hits[0]["text"] == DOCS[0]
    assert any("hybrid_max_chunks" in r.message for r in caplog.records)


def test_with_embeddings_returns_vectors():
    accessor, _ = _accessor()
    hits = asyncio.run(
        accessor.retrieve_ex(fake_embedding(DOCS[1]), 2, with_embeddings=True)
    )
    assert hits[0]["embedding"] == pytest.approx(fake_embedding(DOCS[1]))


def test_plain_retrieve_has_no_embeddings():
    accessor, _ = _accessor()
    hits = asyncio.run(accessor.retrieve(fake_embedding(DOCS[0]), 1))
    assert "embedding" not in hits[0]
