"""chromadb result-shape handling in the Chroma accessor (no server needed).

chromadb >= 0.5 returns result embeddings as numpy arrays; ``x or []`` on
such a value raises, which used to break the hybrid+MMR path.
"""

from __future__ import annotations

import numpy as np
import pytest

from serviette.server.accessors.chroma import _first_or_empty, _or_empty


def test_get_embeddings_as_ndarray():
    arr = np.array([[0.1, 0.2], [0.3, 0.4]])
    out = _or_empty(arr)
    assert out is arr
    assert _or_empty(None) == []


def test_query_embeddings_list_of_ndarray():
    per_query = [np.array([[0.1, 0.2]])]
    assert _first_or_empty(per_query) is per_query[0]
    assert _first_or_empty(None) == []
    assert _first_or_empty([]) == []
    assert _first_or_empty(np.zeros((0, 2))) == []


# ---------------------------------------------------------------------------
# close(): release the HTTP connection pools
# ---------------------------------------------------------------------------


class _FakeServerApi:
    def __init__(self):
        self.cleaned_up = 0

    async def _cleanup(self):
        self.cleaned_up += 1


class _FakeAsyncClient:
    def __init__(self):
        self._server = _FakeServerApi()

    async def get_collection(self, name):
        return object()


def test_close_releases_chroma_http_pools(monkeypatch):
    """chromadb's AsyncClient has no public close; the accessor must still
    release the httpx pools its server API holds (``_cleanup``, the hook
    the client's own ``__aexit__`` uses) instead of dropping references."""

    import asyncio

    import chromadb

    from serviette.config.schema import ChromaConfig
    from serviette.server.accessors.chroma import ChromaAccessor

    created: list[_FakeAsyncClient] = []

    async def fake_http_client(**kwargs):
        client = _FakeAsyncClient()
        created.append(client)
        return client

    monkeypatch.setattr(chromadb, "AsyncHttpClient", fake_http_client)
    accessor = ChromaAccessor(ChromaConfig(type="chroma"))

    async def run():
        await accessor._ensure_collection()
        await accessor.close()
        await accessor.close()  # idempotent

    asyncio.run(run())
    (client,) = created
    assert client._server.cleaned_up == 1
    assert accessor._client is None and accessor._collection is None


def test_close_before_connect_is_a_noop():
    import asyncio

    from serviette.config.schema import ChromaConfig
    from serviette.server.accessors.chroma import ChromaAccessor

    asyncio.run(ChromaAccessor(ChromaConfig(type="chroma")).close())


def test_installed_chromadb_still_exposes_the_cleanup_hook():
    """Guard for the private hook ``close`` relies on: if a chromadb upgrade
    renames it, the pools would silently leak again."""

    async_fastapi = pytest.importorskip("chromadb.api.async_fastapi")

    assert callable(getattr(async_fastapi.AsyncFastAPI, "_cleanup", None))
