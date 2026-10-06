"""DuckDB vector accessor — embedded, zero-setup, in-database vector search.

Reads the table written by the indexer's ``pw.io.duckdb`` snapshot sink:
``(chunk_id, text, metadata, embedding DOUBLE[])``. Retrieval runs entirely
inside DuckDB with ``list_cosine_similarity`` — a vectorized, columnar scan in
native code (no Python loop over rows). For local corpora this answers in
milliseconds without any external service.

Concurrency note: the indexer writes with ``detach_between_batches``,
releasing the single-writer file lock between minibatches; connections here
are short-lived and read-only and retry through those brief lock windows.
Before the indexer's first commit the file (or table) does not exist yet —
that is surfaced as :class:`IndexNotReadyError`, which the server turns into
a friendly HTTP 503 rather than a stack trace.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from serviette.server.accessors.abstract import AsyncVectorAccessor, IndexNotReadyError
from serviette.server.hybrid import KeywordHybridMixin

logger = logging.getLogger(__name__)

_LOCK_HINT = (
    "Could not open the DuckDB database %r (is a streaming indexer holding it "
    "read-write?). DuckDB allows one writer OR multiple readers per file. "
    "Run the indexer with `mode: static` sources, or switch to a client-server "
    "backend (qdrant, pgvector, ...) for concurrent indexing and serving."
)


class DuckDbAccessor(KeywordHybridMixin, AsyncVectorAccessor):
    supports_embeddings = True

    def __init__(self, config) -> None:
        self._path = config.path
        self._table = config.table
        self._init_hybrid(config)

    async def retrieve(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        # DuckDB's Python API is synchronous; run the query off the event loop.
        return await asyncio.to_thread(self._query, embedding, k)

    async def retrieve_ex(
        self,
        embedding: list[float],
        k: int,
        *,
        query_text: str | None = None,
        with_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        vector_hits = await asyncio.to_thread(
            self._query, embedding, k, with_embeddings=with_embeddings
        )
        return await self._fuse(vector_hits, query_text, k, with_embeddings)

    # -- hybrid hooks (KeywordHybridMixin) ------------------------------------

    async def _hybrid_count(self) -> int:
        return await asyncio.to_thread(self._count_rows)

    async def _hybrid_version(self) -> tuple[int, Any]:
        return await asyncio.to_thread(self._version)

    async def _hybrid_fetch_all(self, with_embeddings: bool) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._fetch_all_rows, with_embeddings)

    # A writer flushing with detach_between_batches holds the lock only
    # briefly; ride out that window before declaring the file unreachable.
    _LOCK_RETRIES = 10
    _LOCK_RETRY_DELAY = 0.2  # seconds

    def _connect_with_retry(self):
        import time

        import duckdb

        last_exc: Exception | None = None
        for _ in range(self._LOCK_RETRIES):
            try:
                return duckdb.connect(self._path, read_only=True)
            except duckdb.Error as exc:
                message = str(exc).lower()
                if "does not exist" in message:
                    # The indexer has not created the database file yet.
                    raise IndexNotReadyError(
                        f"DuckDB file {self._path!r} does not exist yet"
                    ) from exc
                if "lock" not in message:
                    raise
                last_exc = exc
                time.sleep(self._LOCK_RETRY_DELAY)
        logger.error(_LOCK_HINT, self._path)
        raise RuntimeError(_LOCK_HINT % (self._path,)) from last_exc

    def _query(
        self, embedding: list[float], k: int, *, with_embeddings: bool = False
    ) -> list[dict[str, Any]]:
        import duckdb

        embedding_col = ", embedding" if with_embeddings else ""
        conn = self._connect_with_retry()
        try:
            rows = conn.execute(
                f"SELECT text, metadata, "
                f"  list_cosine_similarity(embedding, ?::DOUBLE[]) AS score"
                f"{embedding_col} "
                f'FROM "{self._table}" '
                f"WHERE embedding IS NOT NULL "
                f"ORDER BY score DESC NULLS LAST LIMIT ?",
                [list(map(float, embedding)), k],
            ).fetchall()
        except duckdb.CatalogException as exc:
            # The file exists but the table has not been created yet.
            raise IndexNotReadyError(
                f"table {self._table!r} does not exist yet in {self._path!r}"
            ) from exc
        finally:
            conn.close()
        results: list[dict[str, Any]] = []
        for text, metadata, score, *rest in rows:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            hit: dict[str, Any] = {
                "text": text,
                "metadata": metadata or {},
                "score": float(score) if score is not None else 0.0,
            }
            if with_embeddings:
                hit["embedding"] = [float(x) for x in rest[0]]
            results.append(hit)
        return results

    def _count_rows(self) -> int:
        conn = self._connect_with_retry()
        try:
            (count,) = conn.execute(
                f'SELECT count(*) FROM "{self._table}" WHERE embedding IS NOT NULL'
            ).fetchone()
        finally:
            conn.close()
        return int(count)

    def _version(self) -> tuple[int, Any]:
        """``(row count, newest seen_at)``: an edited document is re-emitted
        with a fresher ``seen_at``, so in-place edits that keep the chunk
        count move this where the count alone would not."""

        conn = self._connect_with_retry()
        try:
            count, newest = conn.execute(
                f"SELECT count(*),"
                f"  max(TRY_CAST(json_extract(metadata, '$.seen_at') AS BIGINT)) "
                f'FROM "{self._table}" WHERE embedding IS NOT NULL'
            ).fetchone()
        finally:
            conn.close()
        return int(count), newest

    def _fetch_all_rows(self, with_embeddings: bool) -> list[dict[str, Any]]:
        """Every stored chunk as a BM25 hit dict (read off the event loop).

        Read through the same retrying read-only connections as vector search;
        the mixin rebuilds the BM25 index only when the row count changes —
        fine for the static/slow-moving corpora the embedded backend targets.
        """

        embedding_col = ", embedding" if with_embeddings else ""
        conn = self._connect_with_retry()
        try:
            rows = conn.execute(
                f'SELECT text, metadata{embedding_col} FROM "{self._table}" '
                f"WHERE embedding IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
        hits = []
        for row in rows:
            text, metadata = row[0], row[1]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            hit: dict[str, Any] = {"text": text, "metadata": metadata or {}}
            if with_embeddings:
                hit["embedding"] = [float(x) for x in row[2]]
            hits.append(hit)
        return hits

    # -- document catalog (rag.documents) -------------------------------------

    # The same document identity ``_stats`` counts by.
    _DOCUMENT_SQL = (
        "coalesce("
        "json_extract_string(metadata, '$.path'),"
        "json_extract_string(metadata, '$.id'),"
        "json_extract_string(metadata, '$.name'))"
    )

    async def _catalog_scan(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_documents)

    def _list_documents(self) -> list[dict[str, Any]]:
        conn = self._connect_with_retry()
        try:
            rows = conn.execute(
                f"SELECT {self._DOCUMENT_SQL} AS document, count(*),"
                f"  any_value(metadata) "
                f'FROM "{self._table}" WHERE embedding IS NOT NULL '
                f"GROUP BY document HAVING document IS NOT NULL AND document <> ''"
            ).fetchall()
        finally:
            conn.close()
        documents = []
        for document, chunks, metadata in rows:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            documents.append(
                {"id": document, "metadata": metadata or {}, "chunks": int(chunks)}
            )
        return documents

    async def document_chunks(
        self, document: str, *, with_embeddings: bool = False
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._document_chunks, document, with_embeddings
        )

    def _document_chunks(
        self, document: str, with_embeddings: bool
    ) -> list[dict[str, Any]]:
        embedding_col = ", embedding" if with_embeddings else ""
        conn = self._connect_with_retry()
        try:
            rows = conn.execute(
                f'SELECT text, metadata{embedding_col} FROM "{self._table}" '
                f"WHERE embedding IS NOT NULL AND {self._DOCUMENT_SQL} = ?",
                [document],
            ).fetchall()
        finally:
            conn.close()
        hits = []
        for row in rows:
            text, metadata = row[0], row[1]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            hit: dict[str, Any] = {"text": text, "metadata": metadata or {}}
            if with_embeddings:
                hit["embedding"] = [float(x) for x in row[2]]
            hits.append(hit)
        return hits

    async def stats(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._stats)

    def _stats(self) -> dict[str, Any]:
        conn = self._connect_with_retry()
        try:
            # A document is identified by whatever the source reports: ``path``
            # (fs, s3, sharepoint, pyfilesystem) or ``id`` / ``name`` (gdrive).
            chunks, documents, last = conn.execute(
                f"SELECT count(*),"
                f"  count(DISTINCT coalesce("
                f"    json_extract_string(metadata, '$.path'),"
                f"    json_extract_string(metadata, '$.id'),"
                f"    json_extract_string(metadata, '$.name'))),"
                f"  max(TRY_CAST(json_extract(metadata, '$.seen_at') AS BIGINT)) "
                f'FROM "{self._table}"'
            ).fetchone()
        finally:
            conn.close()
        out: dict[str, Any] = {"chunks": int(chunks), "documents": int(documents)}
        if last:
            out["last_indexed_at"] = int(last)
        return out

    async def close(self) -> None:
        await self._close_hybrid()  # connections themselves are per-query
