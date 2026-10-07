"""Async Milvus accessor using the ``pymilvus`` async client.

Assumes the indexer wrote chunks into a Milvus collection with fields ``text``
(VARCHAR), ``metadata`` (JSON) and ``embedding`` (FLOAT_VECTOR). Search uses the
``COSINE`` metric so the returned distance *is* the cosine similarity.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from serviette.server.accessors.abstract import AsyncVectorAccessor
from serviette.server.hybrid import KeywordHybridMixin


def _entity_metadata(entity: dict) -> dict[str, Any]:
    metadata = entity.get("metadata", {})
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return metadata or {}


class MilvusAccessor(KeywordHybridMixin, AsyncVectorAccessor):
    supports_embeddings = True

    def __init__(self, config) -> None:
        self._uri = config.resolved_uri()
        self._token = config.token
        self._collection = config.collection
        self._client = None
        self._scan_client = None
        self._init_hybrid(config)

    async def _ensure_client(self):
        if self._client is None:
            from pymilvus import AsyncMilvusClient

            kwargs: dict[str, Any] = {"uri": self._uri}
            if self._token:
                kwargs["token"] = self._token
            self._client = AsyncMilvusClient(**kwargs)
            # A collection must be loaded into memory before it can be searched.
            await self._client.load_collection(self._collection)
        return self._client

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
        client = await self._ensure_client()
        output_fields = ["text", "metadata"]
        if with_embeddings:
            output_fields.append("embedding")
        hits = await client.search(
            collection_name=self._collection,
            data=[embedding],
            limit=k,
            output_fields=output_fields,
            search_params={"metric_type": "COSINE"},
        )
        vector_hits: list[dict[str, Any]] = []
        # ``hits`` is a list (one entry per query vector) of lists of hits.
        for hit in hits[0] if hits else []:
            entity = hit.get("entity", hit)
            record: dict[str, Any] = {
                "text": entity.get("text", ""),
                "metadata": _entity_metadata(entity),
                "score": float(hit.get("distance", hit.get("score", 0.0))),
            }
            if with_embeddings and entity.get("embedding") is not None:
                record["embedding"] = [float(x) for x in entity["embedding"]]
            vector_hits.append(record)
        return await self._fuse(vector_hits, query_text, k, with_embeddings)

    # -- hybrid hooks (KeywordHybridMixin) ------------------------------------

    async def _hybrid_count(self) -> int:
        # Not ``get_collection_stats``: its ``row_count`` covers flushed
        # (sealed) segments only. The Pathway writer upserts without flushing
        # — correctly, Milvus discourages per-batch flushes — and a small
        # corpus never reaches the auto-flush thresholds, so that count stays
        # 0 while search sees every row. ``count(*)`` includes growing
        # segments and is Milvus's own recommendation for an exact count.
        client = await self._ensure_client()
        rows = await client.query(
            collection_name=self._collection, filter="", output_fields=["count(*)"]
        )
        return int(rows[0].get("count(*)", 0)) if rows else 0

    async def _hybrid_fetch_all(self, with_embeddings: bool) -> list[dict[str, Any]]:
        # The async client has no full-scan iterator; a short-lived sync client
        # paginates the whole collection off the event loop. ``query_iterator``
        # sidesteps Milvus's offset+limit query window.
        return await asyncio.to_thread(self._scan_all, with_embeddings)

    def _sync_client(self):
        if self._scan_client is None:
            from pymilvus import MilvusClient

            kwargs: dict[str, Any] = {"uri": self._uri}
            if self._token:
                kwargs["token"] = self._token
            self._scan_client = MilvusClient(**kwargs)
        return self._scan_client

    def _scan_all(self, with_embeddings: bool) -> list[dict[str, Any]]:
        client = self._sync_client()
        fields = ["text", "metadata"]
        if with_embeddings:
            fields.append("embedding")
        iterator = client.query_iterator(
            collection_name=self._collection,
            filter="",
            output_fields=fields,
            batch_size=1000,
        )
        hits: list[dict[str, Any]] = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for entity in batch:
                    hit: dict[str, Any] = {
                        "text": entity.get("text", ""),
                        "metadata": _entity_metadata(entity),
                    }
                    if with_embeddings and entity.get("embedding") is not None:
                        hit["embedding"] = [float(x) for x in entity["embedding"]]
                    hits.append(hit)
        finally:
            iterator.close()
        return hits

    async def stats(self) -> dict[str, Any]:
        count = await self._hybrid_count()
        return {"chunks": count}

    async def close(self) -> None:
        await self._close_hybrid()
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._scan_client is not None:
            self._scan_client.close()
            self._scan_client = None
