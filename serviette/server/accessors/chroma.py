"""Async ChromaDB accessor using the ``chromadb`` HTTP client.

The indexer's ``pw.io.chroma`` sink writes each chunk with the text as the
document and the source metadata serialized to a JSON string (Chroma metadata
values must be scalars). The collection must be created with the cosine
distance (``hnsw:space = cosine``); Chroma returns a *distance*, converted here
to a similarity (``1 - distance``).
"""

from __future__ import annotations

import json
from typing import Any

from serviette.server.accessors.abstract import AsyncVectorAccessor
from serviette.server.hybrid import KeywordHybridMixin

_SCAN_PAGE = 1000


def _or_empty(value: Any) -> Any:
    """``value or []`` that survives numpy: chromadb >= 0.5 returns result
    embeddings as ``numpy.ndarray``, whose truth value raises."""

    return [] if value is None else value


def _first_or_empty(value: Any) -> Any:
    """First per-query entry of a query-result field (``None``-safe)."""

    return value[0] if value is not None and len(value) else []


def _chroma_metadata(meta: Any) -> dict[str, Any]:
    raw = (meta or {}).get("metadata")
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


class ChromaAccessor(KeywordHybridMixin, AsyncVectorAccessor):
    supports_embeddings = True

    def __init__(self, config) -> None:
        self._config = config
        self._client = None
        self._collection = None
        self._init_hybrid(config)

    async def _ensure_collection(self):
        if self._collection is None:
            import chromadb

            cfg = self._config
            self._client = await chromadb.AsyncHttpClient(
                host=cfg.host,
                port=cfg.port,
                ssl=cfg.ssl,
                headers=cfg.headers or {},
                tenant=cfg.tenant,
                database=cfg.database,
            )
            self._collection = await self._client.get_collection(cfg.collection)
        return self._collection

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
        collection = await self._ensure_collection()
        include = ["documents", "metadatas", "distances"]
        if with_embeddings:
            include.append("embeddings")
        response = await collection.query(
            query_embeddings=[embedding], n_results=k, include=include
        )
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        embeddings = _first_or_empty(response.get("embeddings")) if with_embeddings else []
        vector_hits: list[dict[str, Any]] = []
        for i, (text, meta, distance) in enumerate(
            zip(documents, metadatas, distances)
        ):
            hit: dict[str, Any] = {
                "text": text or "",
                "metadata": _chroma_metadata(meta),
                "score": 1.0 - float(distance),
            }
            if with_embeddings and i < len(embeddings):
                hit["embedding"] = [float(x) for x in embeddings[i]]
            vector_hits.append(hit)
        return await self._fuse(vector_hits, query_text, k, with_embeddings)

    # -- hybrid hooks (KeywordHybridMixin) ------------------------------------

    async def _hybrid_count(self) -> int:
        collection = await self._ensure_collection()
        return int(await collection.count())

    async def _hybrid_fetch_all(self, with_embeddings: bool) -> list[dict[str, Any]]:
        collection = await self._ensure_collection()
        include = ["documents", "metadatas"]
        if with_embeddings:
            include.append("embeddings")
        hits: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = await collection.get(
                include=include, limit=_SCAN_PAGE, offset=offset
            )
            documents = page.get("documents") or []
            metadatas = page.get("metadatas") or []
            embeddings = _or_empty(page.get("embeddings"))
            if not documents:
                break
            for i, (text, meta) in enumerate(zip(documents, metadatas)):
                hit: dict[str, Any] = {
                    "text": text or "",
                    "metadata": _chroma_metadata(meta),
                }
                if with_embeddings and i < len(embeddings):
                    hit["embedding"] = [float(x) for x in embeddings[i]]
                hits.append(hit)
            if len(documents) < _SCAN_PAGE:
                break
            offset += _SCAN_PAGE
        return hits

    async def stats(self) -> dict[str, Any]:
        return {"chunks": await self._hybrid_count()}

    async def close(self) -> None:
        self._client = None
        self._collection = None
