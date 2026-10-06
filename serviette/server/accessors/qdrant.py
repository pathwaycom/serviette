"""Async Qdrant accessor using ``qdrant-client``.

The indexer creates the collection with a single **named** dense vector slot
(:data:`QDRANT_VECTOR_NAME`, cosine) and the schema-driven ``pw.io.qdrant`` sink
binds the ``embedding`` column to it; the remaining columns (``text``,
``metadata``) live in the point payload. Searches therefore address that slot by
name (``using=``), and the returned score *is* the cosine similarity.

Hybrid mode (``hybrid: true``) adds a keyword leg: an in-process BM25 index
is built by scrolling the collection's texts and fused with the vector
results via reciprocal-rank fusion. The BM25 corpus is capped at
``hybrid_max_chunks`` (in-process indexing does not scale to tens of
millions of chunks); Qdrant-native sparse vectors are the planned
replacement at that scale.
"""

from __future__ import annotations

import logging
from typing import Any

from serviette.server.accessors.abstract import AsyncVectorAccessor, document_key
from serviette.server.hybrid import KeywordHybridMixin

logger = logging.getLogger(__name__)

_SCROLL_PAGE = 1024

# The name of the collection's dense vector slot. Must match the slot
# ``prepare._prepare_qdrant`` creates and the indexer's ``embedding`` column.
QDRANT_VECTOR_NAME = "embedding"


class QdrantAccessor(KeywordHybridMixin, AsyncVectorAccessor):
    supports_embeddings = True

    def __init__(self, config) -> None:
        self._url = config.rest_url()
        self._api_key = config.api_key
        self._collection = config.collection
        self._client = None
        self._init_hybrid(config)

    def _ensure_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)
        return self._client

    @staticmethod
    def _to_hit(point, with_embeddings: bool) -> dict[str, Any]:
        payload = point.payload or {}
        hit: dict[str, Any] = {
            "text": payload.get("text", ""),
            "metadata": payload.get("metadata") or {},
        }
        if with_embeddings:
            # Named-vector collections return a dict {slot: vector}.
            vector = getattr(point, "vector", None)
            if isinstance(vector, dict):
                vector = vector.get(QDRANT_VECTOR_NAME)
            if vector is not None:
                hit["embedding"] = [float(x) for x in vector]
        return hit

    async def retrieve(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        return await self.retrieve_ex(embedding, k)

    async def retrieve_ex(
        self,
        embedding: list[float],
        k: int,
        *,
        query_text: str | None = None,
        with_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        client = self._ensure_client()
        response = await client.query_points(
            collection_name=self._collection,
            query=embedding,
            using=QDRANT_VECTOR_NAME,
            limit=k,
            with_payload=True,
            with_vectors=with_embeddings,
        )
        vector_hits = [
            {**self._to_hit(point, with_embeddings), "score": float(point.score)}
            for point in response.points
        ]
        return await self._fuse(vector_hits, query_text, k, with_embeddings)

    # -- hybrid hooks (KeywordHybridMixin) ------------------------------------

    async def _hybrid_count(self) -> int:
        # Exact: the approximate count can stay put over small changes and
        # would leave the BM25 index stale until the timed refresh.
        client = self._ensure_client()
        return int(
            (await client.count(collection_name=self._collection, exact=True)).count
        )

    async def _hybrid_fetch_all(self, with_embeddings: bool) -> list[dict[str, Any]]:
        client = self._ensure_client()
        hits: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = await client.scroll(
                collection_name=self._collection,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=with_embeddings,
            )
            hits.extend(self._to_hit(point, with_embeddings) for point in points)
            if offset is None:
                break
        return hits

    # -- document catalog (rag.documents) -------------------------------------

    async def _catalog_scan(self) -> list[dict[str, Any]]:
        # Metadata only: the texts are most of the payload and the catalog
        # does not need them.
        client = self._ensure_client()
        documents: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            points, offset = await client.scroll(
                collection_name=self._collection,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=["metadata"],
                with_vectors=False,
            )
            for point in points:
                metadata = (point.payload or {}).get("metadata") or {}
                key = document_key(metadata)
                if key is None:
                    continue
                entry = documents.setdefault(
                    key, {"id": key, "metadata": metadata, "chunks": 0}
                )
                entry["chunks"] += 1
            if offset is None:
                break
        return list(documents.values())

    async def document_chunks(
        self, document: str, *, with_embeddings: bool = False
    ) -> list[dict[str, Any]]:
        # A payload filter on the document identity — evaluated server-side,
        # so only the document's points travel (no payload index is created:
        # the accessor never alters the collection).
        from qdrant_client import models

        client = self._ensure_client()
        selector = models.Filter(
            should=[
                models.FieldCondition(
                    key=f"metadata.{field}", match=models.MatchValue(value=document)
                )
                for field in ("path", "id", "name")
            ]
        )
        hits: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = await client.scroll(
                collection_name=self._collection,
                scroll_filter=selector,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=with_embeddings,
            )
            hits.extend(
                self._to_hit(point, with_embeddings)
                for point in points
                if document_key((point.payload or {}).get("metadata")) == document
            )
            if offset is None:
                break
        return hits

    async def stats(self) -> dict[str, Any]:
        client = self._ensure_client()
        result = await client.count(collection_name=self._collection, exact=False)
        return {"chunks": int(result.count)}

    async def close(self) -> None:
        await self._close_hybrid()
        if self._client is not None:
            await self._client.close()
            self._client = None
