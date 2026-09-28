"""Async Weaviate accessor using the v4 ``weaviate-client``.

The indexer's ``pw.io.weaviate`` sink stores the chunk vector as the object
vector and the remaining columns as properties (``chunk_id``, ``text``, and the
source metadata serialized to a JSON string). Weaviate reports a cosine
*distance*; converted here to a similarity (``1 - distance``).
"""

from __future__ import annotations

import json
from typing import Any

from serviette.server.accessors.abstract import AsyncVectorAccessor
from serviette.server.hybrid import KeywordHybridMixin


def _props_to_hit(props: dict) -> dict[str, Any]:
    props = props or {}
    raw = props.get("metadata")
    metadata = json.loads(raw) if isinstance(raw, str) else (raw or {})
    return {"text": props.get("text", ""), "metadata": metadata}


def _object_vector(obj) -> list[float] | None:
    # v4 returns named vectors as a dict; the default (unnamed) vector lives
    # under the "default" key.
    vector = getattr(obj, "vector", None)
    if isinstance(vector, dict):
        vector = vector.get("default")
    return [float(x) for x in vector] if vector else None


class WeaviateAccessor(KeywordHybridMixin, AsyncVectorAccessor):
    supports_embeddings = True

    def __init__(self, config) -> None:
        self._config = config
        self._client = None
        self._init_hybrid(config)

    async def _ensure_client(self):
        if self._client is None:
            import weaviate
            from weaviate.connect import ConnectionParams

            cfg = self._config
            auth = (
                weaviate.auth.AuthApiKey(cfg.api_key) if cfg.api_key else None
            )
            self._client = weaviate.WeaviateAsyncClient(
                connection_params=ConnectionParams.from_params(
                    http_host=cfg.http_host,
                    http_port=cfg.http_port,
                    http_secure=cfg.http_secure,
                    grpc_host=cfg.grpc_host or cfg.http_host,
                    grpc_port=cfg.grpc_port,
                    grpc_secure=cfg.grpc_secure,
                ),
                auth_client_secret=auth,
            )
            await self._client.connect()
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
        from weaviate.classes.query import MetadataQuery

        collection = client.collections.get(self._config.collection)
        response = await collection.query.near_vector(
            near_vector=list(map(float, embedding)),
            limit=k,
            include_vector=with_embeddings,
            return_metadata=MetadataQuery(distance=True),
        )
        vector_hits: list[dict[str, Any]] = []
        for obj in response.objects:
            distance = obj.metadata.distance if obj.metadata else None
            hit = _props_to_hit(obj.properties)
            hit["score"] = 1.0 - float(distance) if distance is not None else 0.0
            if with_embeddings:
                vec = _object_vector(obj)
                if vec is not None:
                    hit["embedding"] = vec
            vector_hits.append(hit)
        return await self._fuse(vector_hits, query_text, k, with_embeddings)

    # -- hybrid hooks (KeywordHybridMixin) ------------------------------------

    async def _hybrid_count(self) -> int:
        client = await self._ensure_client()
        collection = client.collections.get(self._config.collection)
        result = await collection.aggregate.over_all(total_count=True)
        return int(result.total_count or 0)

    async def _hybrid_fetch_all(self, with_embeddings: bool) -> list[dict[str, Any]]:
        client = await self._ensure_client()
        collection = client.collections.get(self._config.collection)
        hits: list[dict[str, Any]] = []
        # Cursor-based full scan (server-paged), the v4 way to read every object.
        async for obj in collection.iterator(include_vector=with_embeddings):
            hit = _props_to_hit(obj.properties)
            if with_embeddings:
                vec = _object_vector(obj)
                if vec is not None:
                    hit["embedding"] = vec
            hits.append(hit)
        return hits

    async def stats(self) -> dict[str, Any]:
        return {"chunks": await self._hybrid_count()}

    async def close(self) -> None:
        await self._close_hybrid()
        if self._client is not None:
            await self._client.close()
            self._client = None
