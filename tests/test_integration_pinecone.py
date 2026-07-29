"""Integration test: real indexer -> Pinecone Local (Docker) -> Pinecone accessor.

Pinecone Local is Pinecone's official in-memory emulator. It serves each index
on its own port (5081+), embedding that port into the index host, so the
container ports are mapped 1:1 onto the same host ports; the test skips when
they are taken. The index is created up front with the embedder's dimension
(cosine), like on the managed service. Skips when Docker or the ``pinecone``
SDK is unavailable.
"""

from __future__ import annotations

import pytest

pinecone = pytest.importorskip("pinecone")

from serviette.config.schema import PineconeConfig
from serviette.server.accessors.pinecone import PineconeAccessor
from tests.dockerutil import (
    docker_available,
    free_port_range,
    run_container,
    stop_container,
    wait_until,
)
from tests.integration_common import run_backend_scenario

pytestmark = [pytest.mark.integration, pytest.mark.slow]

IMAGE = "ghcr.io/pinecone-io/pinecone-local:latest"
INDEX = "serviette"
API_KEY = "pclocal"  # Pinecone Local accepts any non-empty key
PORT_SPAN = 6  # control plane + a few index data planes


def _control_plane_ready(base: int) -> bool:
    pc = pinecone.Pinecone(api_key=API_KEY, host=f"http://127.0.0.1:{base}")
    pc.list_indexes()
    return True


@pytest.fixture(scope="module")
def pinecone_base_port():
    if not docker_available():
        pytest.skip("Docker is not available")
    # Pinecone Local advertises absolute index ports (base+1, base+2, ...), so
    # the container ports must be published 1:1 — the base is still allocated
    # dynamically, never hardcoded.
    base = free_port_range(PORT_SPAN)
    if base is None:
        pytest.skip(f"Could not find {PORT_SPAN} consecutive free ports")
    container = run_container(
        IMAGE,
        ports={p: p for p in range(base, base + PORT_SPAN)},
        env={"PORT": str(base), "PINECONE_HOST": "localhost"},
    )
    try:
        # No index setup: the indexer's prepare_backend() must create it.
        wait_until(lambda: _control_plane_ready(base), timeout=90)
        yield base
    finally:
        stop_container(container)


def test_pinecone_scenario(pinecone_base_port, tmp_path):
    vdb = {
        "type": "pinecone",
        "index_name": INDEX,
        "api_key": API_KEY,
        "host": f"http://127.0.0.1:{pinecone_base_port}",
    }
    run_backend_scenario(
        tmp_path,
        vdb,
        lambda: PineconeAccessor(PineconeConfig(**vdb)),
    )
