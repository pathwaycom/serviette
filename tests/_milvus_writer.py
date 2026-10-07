"""Subprocess driver for the Milvus integration test.

Every operation that touches the Milvus Lite file runs in its own short-lived
process via this driver, so two processes never hold the embedded server's file
lock at once (Milvus Lite releases the lock only on process exit, not reliably on
``close()``). Results are printed to stdout as JSON.

Usage:
  python -m tests._milvus_writer write    <uri> <collection> <config-json>
  python -m tests._milvus_writer paths    <uri> <collection>
  python -m tests._milvus_writer retrieve <uri> <collection> <query>
  python -m tests._milvus_writer stats    <uri> <collection>
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


def _write(uri: str, collection: str, config_json: str) -> object:
    import pathway as pw

    from serviette.config.schema import ServietteConfig
    from serviette.indexer.graph import build_graph, persistence_config
    from serviette.indexer.prepare import prepare_backend

    config = ServietteConfig.model_validate(json.loads(config_json))
    # The collection (schema + COSINE index) must come from serviette itself.
    prepare_backend(config)
    build_graph(config)
    pw.run(
        persistence_config=persistence_config(config),
        monitoring_level=pw.MonitoringLevel.NONE,
    )
    return {"ok": True}


def _paths(uri: str, collection: str) -> object:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=uri)
    try:
        client.load_collection(collection)
        rows = client.query(collection, filter="", output_fields=["metadata"], limit=1000)
    finally:
        client.close()
    return sorted(Path(r["metadata"]["path"]).name for r in rows)


def _stats(uri: str, collection: str) -> object:
    from serviette.config.schema import MilvusConfig
    from serviette.server.accessors.milvus import MilvusAccessor

    async def go():
        accessor = MilvusAccessor(MilvusConfig(uri=uri, collection=collection))
        try:
            return await accessor.stats()
        finally:
            await accessor.close()

    return asyncio.run(go())


def _retrieve(uri: str, collection: str, query: str) -> object:
    from serviette.config.schema import MilvusConfig
    from serviette.server.accessors.milvus import MilvusAccessor
    from serviette.testing import fake_embedding

    async def go():
        accessor = MilvusAccessor(MilvusConfig(uri=uri, collection=collection))
        try:
            return await accessor.retrieve(fake_embedding(query), k=5)
        finally:
            await accessor.close()

    return asyncio.run(go())


def main() -> None:
    action = sys.argv[1]
    if action == "write":
        result = _write(sys.argv[2], sys.argv[3], sys.argv[4])
    elif action == "paths":
        result = _paths(sys.argv[2], sys.argv[3])
    elif action == "retrieve":
        result = _retrieve(sys.argv[2], sys.argv[3], sys.argv[4])
    elif action == "stats":
        result = _stats(sys.argv[2], sys.argv[3])
    else:  # pragma: no cover
        raise SystemExit(f"unknown action {action!r}")
    print("__RESULT__" + json.dumps(result))


if __name__ == "__main__":
    main()
