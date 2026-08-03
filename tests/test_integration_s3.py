"""Integration test: S3 source (MinIO, Docker) -> real indexer -> DuckDB.

Exercises the ``only_metadata`` S3 source end-to-end: objects in a MinIO
bucket are indexed by the real indexer (bytes fetched on demand by the
S3Fetcher), retrieval runs through the DuckDB accessor, and deleting an
object from the bucket removes its vectors on the next pass.

Requires a Pathway build with s3 ``only_metadata`` support (>= 0.32.1);
skips otherwise, and when Docker or boto3 is unavailable.
"""

from __future__ import annotations

import httpx
import pytest

boto3 = pytest.importorskip("boto3")

from serviette.config.schema import DuckDbConfig
from serviette.server.accessors.duckdb import DuckDbAccessor
from serviette.testing import fake_embedding
from tests.dockerutil import (
    docker_available,
    run_container,
    stop_container,
    wait_until,
)
from tests.integration_common import ALPHA, BETA, run_indexer, write_config

pytestmark = [pytest.mark.integration, pytest.mark.slow]

IMAGE = "minio/minio:latest"
BUCKET = "serviette-docs"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"


def _pathway_supports_s3_only_metadata() -> bool:
    import pathway as pw

    return "only_metadata" in (pw.io.s3.read.__doc__ or "")


def _s3_client(port: int):
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{port}",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(s3={"addressing_style": "path"}),
    )


@pytest.fixture
def minio_port(tcp_port):
    if not docker_available():
        pytest.skip("Docker is not available")
    port = tcp_port
    container = run_container(
        IMAGE,
        ports={port: 9000},
        env={"MINIO_ROOT_USER": ACCESS_KEY, "MINIO_ROOT_PASSWORD": SECRET_KEY},
        command=["server", "/data"],
    )
    try:
        wait_until(
            lambda: httpx.get(
                f"http://127.0.0.1:{port}/minio/health/live"
            ).status_code == 200,
            timeout=90,
        )
        _s3_client(port).create_bucket(Bucket=BUCKET)
        yield port
    finally:
        stop_container(container)


def test_s3_source_scenario(minio_port, tmp_path):
    if not _pathway_supports_s3_only_metadata():
        pytest.skip("This pathway build lacks s3 only_metadata support")

    s3 = _s3_client(minio_port)
    s3.put_object(Bucket=BUCKET, Key="docs/a.txt", Body=ALPHA.encode())
    s3.put_object(Bucket=BUCKET, Key="docs/b.txt", Body=BETA.encode())

    store = tmp_path / "store.duckdb"
    cfg = write_config(
        tmp_path,
        tmp_path,  # docs dir unused; the source section is overridden below
        {"type": "duckdb", "path": str(store), "table": "serviette_embeddings"},
    )
    import yaml

    config = yaml.safe_load(cfg.read_text())
    config["sources"] = [
        {
            "type": "s3",
            "bucket": BUCKET,
            "path": "docs/",
            "access_key": ACCESS_KEY,
            "secret_access_key": SECRET_KEY,
            "endpoint": f"http://127.0.0.1:{minio_port}",
            "region": "us-east-1",  # required with a custom endpoint
            "with_path_style": True,
            "mode": "static",
        }
    ]
    cfg.write_text(yaml.safe_dump(config))

    def retrieve(text: str, k: int):
        import asyncio

        async def go():
            accessor = DuckDbAccessor(
                DuckDbConfig(path=str(store), table="serviette_embeddings")
            )
            try:
                return await accessor.retrieve(fake_embedding(text), k)
            finally:
                await accessor.close()

        return asyncio.run(go())

    # Pass 1: both objects indexed straight from the bucket.
    run_indexer(cfg, tmp_path / "cache")
    results = retrieve(ALPHA, k=2)
    assert results[0]["text"] == ALPHA
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-9)
    assert results[0]["metadata"]["path"].endswith("a.txt")
    assert len(results) == 2

    # Pass 2: deleting the object removes its vectors.
    s3.delete_object(Bucket=BUCKET, Key="docs/b.txt")
    run_indexer(cfg, tmp_path / "cache")
    remaining = {r["metadata"]["path"] for r in retrieve(BETA, k=5)}
    assert remaining == {"docs/a.txt"}
