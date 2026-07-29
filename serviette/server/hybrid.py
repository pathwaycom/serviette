"""Shared in-process BM25 hybrid retrieval for the vector accessors.

A backend opts in by mixing in :class:`KeywordHybridMixin`, calling
``_init_hybrid(config)`` from its ``__init__``, and implementing the two hooks
``_hybrid_count`` and ``_hybrid_fetch_all``. The mixin then fuses the vector
hits with a BM25 keyword search — built in-process from the backend's stored
chunk texts and rebuilt whenever the row count changes — via reciprocal-rank
fusion (see :func:`serviette.server.ranking.rrf_merge`).

In-process BM25 targets corpora up to a few million chunks; above
``hybrid_max_chunks`` the keyword leg is skipped with a one-time warning and
retrieval degrades to pure vector search. Backends whose engine offers native
server-side keyword scoring should prefer that (planned);
backends that cannot enumerate their rows at all (Pinecone) cannot use this
mixin and reject ``hybrid: true`` at startup.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from serviette.server.bm25 import Bm25Index
from serviette.server.ranking import rrf_merge

logger = logging.getLogger(__name__)


class KeywordHybridMixin:
    """In-process BM25 keyword leg + RRF fusion, shared by the accessors.

    Concrete accessors provide the two backend-specific hooks below; everything
    else (the cached BM25 index, the rebuild-on-count-change policy, the size
    guard, the fusion) lives here so every backend behaves identically.
    """

    # Defaults keep the mixin usable even if ``_init_hybrid`` was not called
    # (hybrid simply stays off).
    _hybrid = False
    _hybrid_max_chunks = 5_000_000

    def _init_hybrid(self, config) -> None:
        self._hybrid = getattr(config, "hybrid", False)
        self._hybrid_max_chunks = getattr(config, "hybrid_max_chunks", 5_000_000)
        self._bm25: Bm25Index | None = None
        # The (approximate) row count the current index was built for — compared
        # against fresh counts to decide on a rebuild. Kept separate from
        # ``Bm25Index.doc_count`` (the exact scanned size) so a backend whose
        # count is approximate does not rebuild on every query.
        self._bm25_built_for: int | None = None
        self._bm25_lock = asyncio.Lock()
        self._warned_too_large = False

    # -- backend hooks --------------------------------------------------------

    async def _hybrid_count(self) -> int:
        """A cheap (possibly approximate) row count, for cache invalidation."""
        raise NotImplementedError

    async def _hybrid_fetch_all(self, with_embeddings: bool) -> list[dict[str, Any]]:
        """Every stored chunk as a hit dict (``text``/``metadata`` and, when
        ``with_embeddings``, ``embedding``) — the BM25 corpus."""
        raise NotImplementedError

    # -- shared fusion --------------------------------------------------------

    async def _fuse(
        self,
        vector_hits: list[dict[str, Any]],
        query_text: str | None,
        k: int,
        with_embeddings: bool,
    ) -> list[dict[str, Any]]:
        """Fuse the vector hits with the BM25 leg, when hybrid is enabled.

        Falls back to the vector hits unchanged when hybrid is off, no query
        text was passed, or the keyword leg returned nothing (empty corpus or
        skipped over the size cap).
        """

        if not (self._hybrid and query_text):
            return vector_hits
        keyword_hits = await self._keyword_hits(query_text, k, with_embeddings)
        if not keyword_hits:
            return vector_hits
        return rrf_merge([vector_hits, keyword_hits], k)

    async def _keyword_hits(
        self, query_text: str, k: int, with_embeddings: bool
    ) -> list[dict[str, Any]]:
        count = await self._hybrid_count()
        if count > self._hybrid_max_chunks:
            if not self._warned_too_large:
                self._warned_too_large = True
                logger.warning(
                    "hybrid: ~%d chunks exceeds hybrid_max_chunks=%d — skipping "
                    "the in-process BM25 leg (retrieval stays pure-vector). "
                    "Raise the cap or move to native server-side keyword search "
                    "(planned).",
                    count,
                    self._hybrid_max_chunks,
                )
            return []
        async with self._bm25_lock:
            if (
                self._bm25 is None
                or self._bm25_built_for != count
                or self._bm25.has_embeddings != with_embeddings
            ):
                hits = await self._hybrid_fetch_all(with_embeddings)
                # Tokenizing the whole corpus is CPU-bound; keep it off the loop.
                self._bm25 = await asyncio.to_thread(Bm25Index, hits, with_embeddings)
                self._bm25_built_for = count
            bm25 = self._bm25
        return await asyncio.to_thread(bm25.search, query_text, k)
