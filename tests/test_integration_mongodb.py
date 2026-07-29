"""Integration test: real indexer -> MongoDB Atlas Local (Docker) -> accessor.

``mongodb/mongodb-atlas-local`` bundles mongod + mongot, so the real
``$vectorSearch`` pipeline the production accessor uses runs locally. The
collection and its ``vectorSearch`` index are created up front (as on managed
Atlas); mongot ingests changes asynchronously, so after each indexing pass the
test waits until the search index has caught up with the collection. Skips
when Docker or ``pymongo`` is unavailable.
"""

from __future__ import annotations

import pytest

pymongo = pytest.importorskip("pymongo")

from serviette.config.schema import MongoDbConfig
from serviette.server.accessors.mongodb import MongoDbAccessor
from serviette.testing import fake_embedding
from tests.dockerutil import (
    docker_available,
    run_container,
    stop_container,
    wait_until,
)
from tests.integration_common import ALPHA, run_backend_scenario

pytestmark = [pytest.mark.integration, pytest.mark.slow]

IMAGE = "mongodb/mongodb-atlas-local:latest"
DATABASE = "serviette"
COLLECTION = "serviette_embeddings"
VECTOR_INDEX = "vector_index"


def _client(connection_string: str):
    return pymongo.MongoClient(connection_string, serverSelectionTimeoutMS=2000)


def _ping(connection_string: str) -> bool:
    _client(connection_string).admin.command("ping")
    return True


def _wait_search_synced(connection_string: str) -> None:
    """Block until $vectorSearch sees exactly the documents in the collection
    (mongot ingests mongod changes asynchronously)."""

    coll = _client(connection_string)[DATABASE][COLLECTION]
    probe = fake_embedding(ALPHA)

    def synced() -> bool:
        hits = list(
            coll.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": VECTOR_INDEX,
                            "path": "embedding",
                            "queryVector": probe,
                            "numCandidates": 100,
                            "limit": 10,
                        }
                    },
                    {"$count": "n"},
                ]
            )
        )
        found = hits[0]["n"] if hits else 0
        return found == coll.count_documents({})

    wait_until(synced, timeout=120)


@pytest.fixture
def mongodb_uri(tcp_port):
    if not docker_available():
        pytest.skip("Docker is not available")
    port = tcp_port
    container = run_container(IMAGE, ports={port: 27017}, env={})
    uri = f"mongodb://127.0.0.1:{port}/?directConnection=true"
    try:
        # No collection/index setup: the indexer's prepare_backend() must
        # create both; _wait_search_synced then waits for queryability.
        wait_until(lambda: _ping(uri), timeout=180)
        yield uri
    finally:
        stop_container(container)


def test_mongodb_scenario(mongodb_uri, tmp_path):
    vdb = {
        "type": "mongodb",
        "connection_string": mongodb_uri,
        "database": DATABASE,
        "collection": COLLECTION,
        "vector_index": VECTOR_INDEX,
    }
    run_backend_scenario(
        tmp_path,
        vdb,
        lambda: MongoDbAccessor(MongoDbConfig(**vdb)),
        after_index=lambda: _wait_search_synced(mongodb_uri),
        score_abs=1e-3,  # ANN + score renormalization
        make_hybrid_accessor=lambda: MongoDbAccessor(
            MongoDbConfig(**{**vdb, "hybrid": True})
        ),
    )
