"""Integration test: real indexer -> embedded DuckDB -> DuckDB accessor.

No external service (DuckDB is embedded), so this runs everywhere — it is the
same scenario the Docker-backed integration tests run against client-server
databases. Marked ``slow`` only.
"""

from __future__ import annotations

import pytest

from serviette.config.schema import DuckDbConfig
from serviette.server.accessors.duckdb import DuckDbAccessor
from tests.integration_common import run_backend_scenario

pytestmark = pytest.mark.slow

TABLE = "serviette_embeddings"


def test_duckdb_scenario(tmp_path):
    store = tmp_path / "store.duckdb"
    run_backend_scenario(
        tmp_path,
        {"type": "duckdb", "path": str(store), "table": TABLE},
        lambda: DuckDbAccessor(DuckDbConfig(path=str(store), table=TABLE)),
        score_abs=1e-9,  # exact in-database cosine, no ANN approximation
    )
