"""Async MongoDB Atlas Vector Search accessor using ``pymongo``'s async API.

The indexer's ``pw.io.mongodb`` snapshot sink stores one document per chunk
(``chunk_id``, ``text``, ``metadata``, ``embedding`` as a number array).
Retrieval runs an Atlas ``$vectorSearch`` aggregation against the configured
``vectorSearch`` index (create it on the ``embedding`` path with
``numDimensions`` matching the embedder and the cosine similarity).

Atlas normalizes the cosine score to ``(1 + cosine) / 2`` ∈ [0, 1]; converted
back to a plain cosine similarity here to match the accessor contract.
"""

from __future__ import annotations

import json
from typing import Any

from serviette.server.accessors.abstract import AsyncVectorAccessor
from serviette.server.hybrid import KeywordHybridMixin

# Atlas recommends ~10-20x the limit for the ANN candidate pool.
_CANDIDATE_FACTOR = 15


def _doc_metadata(doc: dict) -> dict[str, Any]:
    # The sink stores the pw.Json metadata as a JSON string in BSON.
    metadata = doc.get("metadata")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return metadata or {}


class MongoDbAccessor(KeywordHybridMixin, AsyncVectorAccessor):
    supports_embeddings = True

    def __init__(self, config) -> None:
        self._config = config
        self._client = None
        self._init_hybrid(config)

    def _ensure_collection(self):
        if self._client is None:
            from pymongo import AsyncMongoClient

            self._client = AsyncMongoClient(self._config.connection_string)
        return self._client[self._config.database][self._config.collection]

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
        collection = self._ensure_collection()
        projection: dict[str, Any] = {
            "_id": 0,
            "text": 1,
            "metadata": 1,
            "score": {"$meta": "vectorSearchScore"},
        }
        if with_embeddings:
            projection["embedding"] = 1
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self._config.vector_index,
                    "path": "embedding",
                    "queryVector": list(map(float, embedding)),
                    "numCandidates": max(k * _CANDIDATE_FACTOR, 100),
                    "limit": k,
                }
            },
            {"$project": projection},
        ]
        vector_hits: list[dict[str, Any]] = []
        async for doc in await collection.aggregate(pipeline):
            hit: dict[str, Any] = {
                "text": doc.get("text", ""),
                "metadata": _doc_metadata(doc),
                # Undo Atlas's (1 + cosine) / 2 normalization.
                "score": 2.0 * float(doc.get("score", 0.5)) - 1.0,
            }
            if with_embeddings and doc.get("embedding") is not None:
                hit["embedding"] = [float(x) for x in doc["embedding"]]
            vector_hits.append(hit)
        return await self._fuse(vector_hits, query_text, k, with_embeddings)

    # -- hybrid hooks (KeywordHybridMixin) ------------------------------------

    async def _hybrid_count(self) -> int:
        collection = self._ensure_collection()
        return int(await collection.estimated_document_count())

    async def _hybrid_fetch_all(self, with_embeddings: bool) -> list[dict[str, Any]]:
        collection = self._ensure_collection()
        projection: dict[str, Any] = {"_id": 0, "text": 1, "metadata": 1}
        if with_embeddings:
            projection["embedding"] = 1
        hits: list[dict[str, Any]] = []
        async for doc in collection.find({}, projection):
            hit: dict[str, Any] = {
                "text": doc.get("text", ""),
                "metadata": _doc_metadata(doc),
            }
            if with_embeddings and doc.get("embedding") is not None:
                hit["embedding"] = [float(x) for x in doc["embedding"]]
            hits.append(hit)
        return hits

    async def stats(self) -> dict[str, Any]:
        return {"chunks": await self._hybrid_count()}

    async def close(self) -> None:
        await self._close_hybrid()
        if self._client is not None:
            await self._client.close()
            self._client = None
