"""Integration test: real indexer -> Weaviate (Docker) -> Weaviate accessor.

The collection must pre-exist; created here with no server-side vectorizer
(serviette supplies vectors) and explicit TEXT properties matching the sink's
columns. Skips when Docker or ``weaviate-client`` is unavailable.
"""

from __future__ import annotations

import pytest

weaviate = pytest.importorskip("weaviate")

from serviette.config.schema import WeaviateConfig
from serviette.server.accessors.weaviate import WeaviateAccessor
from tests.dockerutil import (
    docker_available,
    run_container,
    stop_container,
    wait_until,
)
from tests.integration_common import run_backend_scenario

pytestmark = [pytest.mark.integration, pytest.mark.slow]

IMAGE = "cr.weaviate.io/semitechnologies/weaviate:1.28.2"
COLLECTION = "ServietteEmbeddings"


def _ready(http_port: int, grpc_port: int) -> bool:
    # No collection setup: the indexer's prepare_backend() must create it.
    client = weaviate.connect_to_local(
        host="127.0.0.1", port=http_port, grpc_port=grpc_port
    )
    try:
        return client.is_ready()
    finally:
        client.close()


@pytest.fixture
def weaviate_ports(tcp_port_factory):
    if not docker_available():
        pytest.skip("Docker is not available")
    http_port, grpc_port = tcp_port_factory(), tcp_port_factory()
    container = run_container(
        IMAGE,
        ports={http_port: 8080, grpc_port: 50051},
        env={
            "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
            "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
            "DEFAULT_VECTORIZER_MODULE": "none",
            "CLUSTER_HOSTNAME": "node1",
        },
    )
    try:
        wait_until(lambda: _ready(http_port, grpc_port), timeout=120)
        yield http_port, grpc_port
    finally:
        stop_container(container)


def test_weaviate_scenario(weaviate_ports, tmp_path):
    http_port, grpc_port = weaviate_ports
    vdb = {
        "type": "weaviate",
        "http_host": "127.0.0.1",
        "http_port": http_port,
        "grpc_port": grpc_port,
        "collection": COLLECTION,
    }
    run_backend_scenario(
        tmp_path,
        vdb,
        lambda: WeaviateAccessor(WeaviateConfig(**vdb)),
        make_hybrid_accessor=lambda: WeaviateAccessor(
            WeaviateConfig(**{**vdb, "hybrid": True})
        ),
    )
