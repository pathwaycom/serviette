"""Async pgvector accessor using ``asyncpg``.

Assumes the indexer wrote chunks into a table whose columns include ``text``,
``metadata`` (jsonb) and ``embedding`` (a ``vector(n)`` column from the pgvector
extension). Retrieval uses pgvector's cosine-distance operator ``<=>`` and
converts distance to a cosine *similarity* (``1 - distance``) for the response.
"""

from __future__ import annotations

import json
from typing import Any

from serviette.server.accessors.abstract import AsyncVectorAccessor
from serviette.server.hybrid import KeywordHybridMixin


def _to_vector_literal(embedding: list[float]) -> str:
    """Render an embedding as a pgvector text literal: ``[1,2,3]``."""

    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _parse_metadata(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else (value or {})


def _parse_embedding(value: Any) -> list[float]:
    # asyncpg returns a pgvector ``vector`` column as its text literal
    # ``[1,2,3]`` (valid JSON) unless a codec is registered.
    if isinstance(value, str):
        value = json.loads(value)
    return [float(x) for x in value]


class PgVectorAccessor(KeywordHybridMixin, AsyncVectorAccessor):
    supports_embeddings = True

    def __init__(self, config) -> None:
        self._dsn = config.connection_string
        self._table = config.table
        self._pool = None
        self._init_hybrid(config)

    async def _ensure_pool(self):
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(dsn=self._dsn)
        return self._pool

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
        pool = await self._ensure_pool()
        vec = _to_vector_literal(embedding)
        embedding_col = ", embedding" if with_embeddings else ""
        # ``embedding <=> $1`` is cosine distance (0 = identical). Order ascending
        # by distance and convert to similarity in the projection.
        query = (
            f"SELECT text, metadata, 1 - (embedding <=> $1::vector) AS score"
            f"{embedding_col} "
            f"FROM {self._table} "
            f"ORDER BY embedding <=> $1::vector ASC "
            f"LIMIT $2"
        )
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, vec, k)
        vector_hits: list[dict[str, Any]] = []
        for row in rows:
            hit: dict[str, Any] = {
                "text": row["text"],
                "metadata": _parse_metadata(row["metadata"]),
                "score": float(row["score"]),
            }
            if with_embeddings:
                hit["embedding"] = _parse_embedding(row["embedding"])
            vector_hits.append(hit)
        return await self._fuse(vector_hits, query_text, k, with_embeddings)

    # -- hybrid hooks (KeywordHybridMixin) ------------------------------------

    async def _hybrid_count(self) -> int:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            return int(await conn.fetchval(f"SELECT count(*) FROM {self._table}"))

    async def _hybrid_version(self) -> tuple[int, Any]:
        # Newest seen_at moves on in-place edits the count cannot see.
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT count(*) AS chunks, "
                f"max((metadata->>'seen_at')::bigint) AS newest FROM {self._table}"
            )
        return int(row["chunks"]), row["newest"]

    async def _hybrid_fetch_all(self, with_embeddings: bool) -> list[dict[str, Any]]:
        pool = await self._ensure_pool()
        embedding_col = ", embedding" if with_embeddings else ""
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT text, metadata{embedding_col} FROM {self._table}"
            )
        hits: list[dict[str, Any]] = []
        for row in rows:
            hit: dict[str, Any] = {
                "text": row["text"],
                "metadata": _parse_metadata(row["metadata"]),
            }
            if with_embeddings:
                hit["embedding"] = _parse_embedding(row["embedding"])
            hits.append(hit)
        return hits

    async def stats(self) -> dict[str, Any]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            # Document identity per source type: path (fs, s3, sharepoint,
            # pyfilesystem) or id / name (gdrive).
            row = await conn.fetchrow(
                f"SELECT count(*) AS chunks,"
                f"  count(DISTINCT coalesce(metadata->>'path', metadata->>'id',"
                f"    metadata->>'name')) AS documents,"
                f"  max((metadata->>'seen_at')::bigint) AS last_indexed_at "
                f"FROM {self._table}"
            )
        out: dict[str, Any] = {
            "chunks": int(row["chunks"]),
            "documents": int(row["documents"]),
        }
        if row["last_indexed_at"]:
            out["last_indexed_at"] = int(row["last_indexed_at"])
        return out

    async def close(self) -> None:
        await self._close_hybrid()
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
