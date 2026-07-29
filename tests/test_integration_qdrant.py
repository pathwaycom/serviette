"""Integration test: real indexer -> Qdrant (Docker) -> Qdrant accessor.

The indexer's ``prepare`` step creates the collection with a named cosine
dense slot (the schema-driven sink no longer auto-creates it), so no schema
setup is needed here. Skips when Docker or ``qdrant-client`` is unavailable.
"""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("qdrant_client")

from serviette.config.schema import QdrantConfig
from serviette.server.accessors.qdrant import QdrantAccessor
from tests.dockerutil import (
    docker_available,
    run_container,
    stop_container,
    wait_until,
)
from tests.integration_common import run_backend_scenario

pytestmark = [pytest.mark.integration, pytest.mark.slow]

IMAGE = "qdrant/qdrant:latest"
COLLECTION = "serviette_embeddings"


@pytest.fixture
def qdrant_ports(tcp_port_factory):
    if not docker_available():
        pytest.skip("Docker is not available")
    rest_port, grpc_port = tcp_port_factory(), tcp_port_factory()
    container = run_container(
        IMAGE, ports={rest_port: 6333, grpc_port: 6334}, env={}
    )
    try:
        wait_until(
            lambda: httpx.get(f"http://127.0.0.1:{rest_port}/readyz").status_code == 200,
            timeout=90,
        )
        yield rest_port, grpc_port
    finally:
        stop_container(container)


def test_qdrant_scenario(qdrant_ports, tmp_path):
    rest_port, grpc_port = qdrant_ports
    vdb = {
        "type": "qdrant",
        "host": "127.0.0.1",
        "rest_port": rest_port,
        "grpc_port": grpc_port,
        "collection": COLLECTION,
    }
    run_backend_scenario(
        tmp_path,
        vdb,
        lambda: QdrantAccessor(QdrantConfig(**vdb)),
    )
