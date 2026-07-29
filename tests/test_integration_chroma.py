"""Integration test: real indexer -> ChromaDB (Docker) -> Chroma accessor.

The collection must pre-exist (the connector never creates it); it is created
here with the cosine ``hnsw:space`` the accessor's score conversion assumes.
Skips when Docker or the ``chromadb`` client is unavailable.
"""

from __future__ import annotations

import httpx
import pytest

chromadb = pytest.importorskip("chromadb")

from serviette.config.schema import ChromaConfig
from serviette.server.accessors.chroma import ChromaAccessor
from tests.dockerutil import (
    docker_available,
    run_container,
    stop_container,
    wait_until,
)
from tests.integration_common import run_backend_scenario

pytestmark = [pytest.mark.integration, pytest.mark.slow]

IMAGE = "chromadb/chroma:1.5.9"
COLLECTION = "serviette_embeddings"


def _heartbeat(port: int) -> bool:
    return httpx.get(f"http://127.0.0.1:{port}/api/v2/heartbeat").status_code == 200


@pytest.fixture
def chroma_port(tcp_port):
    if not docker_available():
        pytest.skip("Docker is not available")
    port = tcp_port
    container = run_container(IMAGE, ports={port: 8000}, env={})
    try:
        # No collection setup: the indexer's prepare_backend() must create it.
        wait_until(lambda: _heartbeat(port), timeout=90)
        yield port
    finally:
        stop_container(container)


def test_chroma_scenario(chroma_port, tmp_path):
    vdb = {
        "type": "chroma",
        "host": "127.0.0.1",
        "port": chroma_port,
        "collection": COLLECTION,
    }
    run_backend_scenario(
        tmp_path,
        vdb,
        lambda: ChromaAccessor(ChromaConfig(**vdb)),
        make_hybrid_accessor=lambda: ChromaAccessor(ChromaConfig(**{**vdb, "hybrid": True})),
    )
