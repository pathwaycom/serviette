"""DuckDB startup states are temporary unavailability, not broken requests."""

from __future__ import annotations

import duckdb
import pytest
from fastapi.testclient import TestClient

from serviette.config.schema import DuckDbConfig, EmbedderConfig, ServietteConfig
from serviette.server.accessors.abstract import IndexNotReadyError
from serviette.server.accessors.duckdb import DuckDbAccessor
from serviette.server.main import create_app
from tests.conftest import fake_embedding, write_duckdb_rows


@pytest.fixture(params=["missing_file", "missing_table"])
def uninitialized_store(tmp_path, request):
    path = tmp_path / "starting.duckdb"
    if request.param == "missing_table":
        duckdb.connect(str(path)).close()
    return path


@pytest.mark.parametrize(
    "method,args,kwargs",
    [
        ("_query", ([1.0], 2), {}),
        ("_query", ([1.0], 2), {"with_embeddings": True}),
        ("_count_rows", (), {}),
        ("_version", (), {}),
        ("_fetch_all_rows", (False,), {}),
        ("_fetch_all_rows", (True,), {}),
        ("_list_documents", (), {}),
        ("_document_chunks", ("document.txt", False), {}),
        ("_document_chunks", ("document.txt", True), {}),
        ("_stats", (), {}),
    ],
)
def test_all_duckdb_reads_report_an_uninitialized_index(
    uninitialized_store, method, args, kwargs
):
    accessor = DuckDbAccessor(DuckDbConfig(path=str(uninitialized_store)))
    with pytest.raises(IndexNotReadyError, match="does not exist yet"):
        getattr(accessor, method)(*args, **kwargs)


@pytest.mark.parametrize(
    "endpoint", ["/api/v1/documents", "/api/v1/retrieve", "/retrieve", "/api/v1/rag", "/rag"]
)
def test_data_endpoints_report_index_startup_with_retry_after(
    uninitialized_store, mock_server_embedder, mock_llm, endpoint
):
    config = ServietteConfig(
        vector_db=DuckDbConfig(path=str(uninitialized_store), hybrid=True),
        embedder=EmbedderConfig(type="mock"),
    )
    app = create_app(config, embedder=mock_server_embedder, llm=mock_llm)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = (
            client.get(endpoint)
            if endpoint.endswith("/documents")
            else client.post(endpoint, json={"query": "hello"})
        )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert "not ready" in response.json()["detail"]
    assert str(uninitialized_store) in response.json()["reason"]


def test_observability_endpoints_remain_available_during_index_startup(
    uninitialized_store, mock_server_embedder
):
    config = ServietteConfig(
        vector_db=DuckDbConfig(path=str(uninitialized_store)),
        embedder=EmbedderConfig(type="mock"),
    )
    with TestClient(create_app(config, embedder=mock_server_embedder)) as client:
        for endpoint in ("/api/v1/health", "/health", "/api/v1/config"):
            assert client.get(endpoint).status_code == 200
        response = client.get("/api/v1/stats")
    assert response.status_code == 200
    assert response.json() == {"backend": "duckdb", "stats_available": False}


async def test_catalog_recovers_after_the_indexer_creates_the_table(uninitialized_store):
    accessor = DuckDbAccessor(DuckDbConfig(path=str(uninitialized_store)))
    with pytest.raises(IndexNotReadyError):
        await accessor.list_documents()
    text = "A newly indexed document."
    write_duckdb_rows(uninitialized_store, [{
        "id": "chunk-1",
        "text": text,
        "metadata": {"path": "document.txt"},
        "embedding": fake_embedding(text),
    }])
    catalog = await accessor.list_documents()
    assert [(entry["id"], entry["chunks"]) for entry in catalog] == [("document.txt", 1)]
    chunks = await accessor.document_chunks("document.txt", with_embeddings=True)
    assert [chunk["text"] for chunk in chunks] == [text]
    hits = await accessor.retrieve_ex(fake_embedding(text), 2, query_text=text)
    assert [hit["text"] for hit in hits] == [text]
    await accessor.close()


def test_an_empty_initialized_table_is_queryable(tmp_path, mock_server_embedder):
    path = tmp_path / "empty.duckdb"
    write_duckdb_rows(path, [])
    config = ServietteConfig(
        vector_db=DuckDbConfig(path=str(path), hybrid=True),
        embedder=EmbedderConfig(type="mock"),
    )
    with TestClient(create_app(config, embedder=mock_server_embedder)) as client:
        catalog = client.get("/api/v1/documents")
        results = client.post("/api/v1/retrieve", json={"query": "hello"})
    assert catalog.status_code == results.status_code == 200
    assert catalog.json() == {"total": 0, "matched": 0, "truncated": False, "documents": []}
    assert results.json() == {"results": []}


@pytest.mark.parametrize("method,args", [("_query", ([1.0], 2)), ("_list_documents", ())])
def test_other_catalog_errors_are_not_reported_as_index_startup(tmp_path, method, args):
    path = tmp_path / "invalid-view.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE MACRO serviette_test_metadata(value) AS value")
        conn.execute(
            "CREATE VIEW serviette_embeddings AS SELECT 'text' AS text, "
            "serviette_test_metadata('{}') AS metadata, [1.0] AS embedding"
        )
        conn.execute("DROP MACRO serviette_test_metadata")
    accessor = DuckDbAccessor(DuckDbConfig(path=str(path)))
    with pytest.raises(duckdb.CatalogException, match="serviette_test_metadata"):
        getattr(accessor, method)(*args)
