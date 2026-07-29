"""Async Pinecone accessor using the ``pinecone`` SDK's asyncio support.

The indexer's ``pw.io.pinecone`` sink upserts each chunk keyed by ``chunk_id``
with ``text`` and the JSON-serialized source metadata as record metadata. With
a cosine-metric index the returned score *is* the cosine similarity.
"""

from __future__ import annotations

import json
import os
from typing import Any

from serviette.server.accessors.abstract import AsyncVectorAccessor


class PineconeAccessor(AsyncVectorAccessor):
    # MMR works (queries can return vectors); in-process BM25 hybrid does not —
    # Pinecone has no scan-all-vectors API to build the keyword corpus from.
    supports_embeddings = True

    def __init__(self, config) -> None:
        if getattr(config, "hybrid", False):
            raise ValueError(
                "hybrid retrieval is not supported on the Pinecone backend: it "
                "has no API to enumerate all stored vectors, so an in-process "
                "BM25 corpus cannot be built. Native hybrid needs a separate "
                "sparse index fed by the indexer (planned); "
                "for in-process hybrid today use duckdb, qdrant, pgvector, "
                "milvus, weaviate, chroma or mongodb."
            )
        self._config = config
        self._pc = None
        self._index = None

    async def _ensure_index(self):
        if self._index is None:
            from pinecone import PineconeAsyncio

            cfg = self._config
            api_key = cfg.api_key or os.environ.get("PINECONE_API_KEY")
            # ``host`` (if set) points at the control plane — e.g. Pinecone
            # Local. The data-plane host differs per index even locally, so it
            # is always resolved through describe_index.
            self._pc = PineconeAsyncio(api_key=api_key, host=cfg.host)
            description = await self._pc.describe_index(cfg.index_name)
            index_host = description.host
            # Pinecone Local serves indexes over plain HTTP but reports their
            # hosts as https; mirror the control plane's scheme.
            if cfg.host and cfg.host.startswith("http://"):
                index_host = index_host.replace("https://", "http://", 1)
            self._index = self._pc.IndexAsyncio(host=index_host)
        return self._index

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
        # ``query_text`` is accepted for interface parity but unused: Pinecone
        # rejects hybrid at startup, so this path is always pure-vector.
        index = await self._ensure_index()
        response = await index.query(
            vector=list(map(float, embedding)),
            top_k=k,
            namespace=self._config.namespace,
            include_metadata=True,
            include_values=with_embeddings,
        )
        results: list[dict[str, Any]] = []
        for match in response.matches:
            meta = dict(match.metadata or {})
            raw = meta.get("metadata")
            metadata = json.loads(raw) if isinstance(raw, str) else (raw or {})
            hit: dict[str, Any] = {
                "text": meta.get("text", ""),
                "metadata": metadata,
                "score": float(match.score),
            }
            if with_embeddings and getattr(match, "values", None):
                hit["embedding"] = [float(x) for x in match.values]
            results.append(hit)
        return results

    async def stats(self) -> dict[str, Any]:
        index = await self._ensure_index()
        description = await index.describe_index_stats()
        namespace = self._config.namespace
        if namespace:
            ns = (description.namespaces or {}).get(namespace)
            count = ns.vector_count if ns else 0
        else:
            count = description.total_vector_count
        return {"chunks": int(count or 0)}

    async def close(self) -> None:
        if self._index is not None:
            await self._index.close()
            self._index = None
        if self._pc is not None:
            await self._pc.close()
            self._pc = None
