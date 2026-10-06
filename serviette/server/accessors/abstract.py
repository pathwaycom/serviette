"""Abstract base class for async vector-DB retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import Self


class IndexNotReadyError(RuntimeError):
    """The vector store exists in config but has no data yet.

    Raised while the indexer is still starting or has not committed its
    first batch (e.g. the DuckDB file or the target table/collection does
    not exist yet). The server maps it to HTTP 503 with a friendly message
    instead of a stack trace."""


def document_key(metadata: dict[str, Any] | None) -> str | None:
    """The identity of a chunk's source document, as the source reports it:
    ``path`` (fs, s3, sharepoint, pyfilesystem) or ``id`` / ``name`` (gdrive).
    The same rule the backends' ``stats`` use to count documents."""

    if not metadata:
        return None
    for field in ("path", "id", "name"):
        value = metadata.get(field)
        if value is not None and value != "":
            return str(value)
    return None


class AsyncVectorAccessor(ABC):
    """Retrieve the nearest chunks for a query embedding from a vector store.

    Implementations are async so the network-bound call to the vector DB does
    not block the FastAPI event loop.
    """

    # Whether ``retrieve_ex(..., with_embeddings=True)`` returns hit
    # embeddings (needed by MMR). Backends that can cheaply return stored
    # vectors flip this to True and honor the flag.
    supports_embeddings = False

    # Whether the backend can enumerate its documents (``list_documents`` /
    # ``document_chunks``) — what ``rag.documents`` needs. Backends that
    # cannot list their rows at all (Pinecone) leave this False.
    supports_catalog = False

    async def list_documents(self) -> list[dict[str, Any]]:
        """Every indexed document: ``{"id": str, "metadata": dict,
        "chunks": int}``, where ``id`` is :func:`document_key` of its chunks
        and ``metadata`` is that of one of them."""

        raise NotImplementedError

    async def document_chunks(
        self, document: str, *, with_embeddings: bool = False
    ) -> list[dict[str, Any]]:
        """All chunks of the document with the given ``id``, as hit dicts
        without a score. The store keeps no chunk order, so none is implied."""

        raise NotImplementedError

    @abstractmethod
    async def retrieve(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        """Return up to ``k`` results ordered by descending similarity.

        Each result is a dict ``{"text": str, "metadata": dict, "score": float}``
        where ``score`` is a cosine similarity in ``[-1, 1]`` (higher is closer).
        """

    async def retrieve_ex(
        self,
        embedding: list[float],
        k: int,
        *,
        query_text: str | None = None,
        with_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """:meth:`retrieve` with optional extras; the server always calls this.

        ``query_text`` lets hybrid-capable backends run a keyword search next
        to the vector one; ``with_embeddings`` asks for an ``"embedding"`` key
        on each hit. The default implementation ignores both and delegates,
        so plain backends need not change.
        """

        return await self.retrieve(embedding, k)

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying connections / pools."""

    async def stats(self) -> dict[str, Any]:
        """Lightweight backend statistics for observability.

        Best-effort keys: ``chunks`` (row/point count), ``documents``
        (distinct source objects, where cheap), ``last_indexed_at`` (unix
        seconds, where cheap). Returns ``{}`` when the backend cannot answer
        cheaply; must never raise for routine unavailability.
        """

        return {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
