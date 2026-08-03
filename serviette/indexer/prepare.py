"""Backend preparation: create the targets the output connectors don't.

Pathway's output connectors auto-create their target only for DuckDB
(``init_mode``) and Qdrant (collection on first write). Every other backend
expects the table / collection / index to exist — so the indexer prepares
them here at startup, and a first run against a fresh database just works:

- **pgvector** — ``CREATE EXTENSION vector``, the table, and an HNSW cosine
  index on ``embedding``.
- **milvus** — collection ``(chunk_id VARCHAR pk, text VARCHAR, metadata
  JSON, embedding FLOAT_VECTOR(dim))`` with a COSINE AUTOINDEX.
- **chroma** — collection with the cosine ``hnsw:space``.
- **weaviate** — collection with no server-side vectorizer and TEXT
  properties matching the sink's columns.
- **pinecone** — serverless index (cosine) with the embedder's dimension.
- **mongodb** — collection plus the Atlas ``vectorSearch`` index on the
  ``embedding`` path.

All operations are create-if-missing (idempotent, race-tolerant). Vector
dimension resolves from ``vector_db.embedding_dimension`` when set, otherwise
from the embedder itself (``BaseEmbedder.get_embedding_dimension()`` — one
probe embedding call).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from serviette.config.schema import (
    ChromaConfig,
    DuckDbConfig,
    MilvusConfig,
    MongoDbConfig,
    PgVectorConfig,
    PineconeConfig,
    QdrantConfig,
    ServietteConfig,
    WeaviateConfig,
)

logger = logging.getLogger(__name__)


def resolve_embedding_dimension(config: ServietteConfig) -> int:
    explicit = getattr(config.vector_db, "embedding_dimension", None)
    if explicit:
        return int(explicit)
    assert config.embedder is not None  # enforced by for_indexer()
    if config.embedder.type == "mock":
        from serviette.testing import EMBED_DIM

        return EMBED_DIM
    from serviette.indexer.graph import build_xpack_embedder

    logger.info("Resolving the embedding dimension with a probe embedding call")
    embedder = build_xpack_embedder(config.embedder)
    # xpack embedders subclass pw.UDF and add get_embedding_dimension().
    return int(embedder.get_embedding_dimension())  # type: ignore[attr-defined]


def prepare_backend(config: ServietteConfig) -> None:
    """Create the vector-DB target for ``config`` if it does not exist."""

    vdb = config.vector_db
    assert vdb is not None  # enforced by for_indexer() before we are called
    if vdb.type == "qdrant":
        _prepare_qdrant(config)
    elif vdb.type == "duckdb":
        _prepare_duckdb(config)
    elif vdb.type == "pgvector":
        _prepare_pgvector(config)
    elif vdb.type == "milvus":
        _prepare_milvus(config)
    elif vdb.type == "chroma":
        _prepare_chroma(config)
    elif vdb.type == "weaviate":
        _prepare_weaviate(config)
    elif vdb.type == "pinecone":
        _prepare_pinecone(config)
    elif vdb.type == "mongodb":
        _prepare_mongodb(config)


def _prepare_pgvector(config: ServietteConfig) -> None:
    import asyncpg

    vdb = config.vector_db
    # prepare_backend dispatches on vdb.type, so the narrow type holds.
    assert isinstance(vdb, PgVectorConfig)
    dimension = resolve_embedding_dimension(config)

    async def go() -> None:
        conn = await asyncpg.connect(vdb.connection_string)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"CREATE TABLE IF NOT EXISTS {vdb.table} ("
                f"  chunk_id  text PRIMARY KEY,"
                f"  text      text,"
                f"  metadata  jsonb,"
                f"  embedding vector({dimension})"
                f")"
            )
            try:
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {vdb.table}_embedding_hnsw "
                    f"ON {vdb.table} USING hnsw (embedding vector_cosine_ops)"
                )
            except asyncpg.PostgresError as exc:
                # HNSW needs pgvector >= 0.5; retrieval still works unindexed.
                logger.warning("Could not create the HNSW index: %s", exc)
        finally:
            await conn.close()

    asyncio.run(go())
    logger.info("pgvector: table %r ready (dimension %d)", vdb.table, dimension)


def _prepare_milvus(config: ServietteConfig) -> None:
    from pymilvus import DataType, MilvusClient

    vdb = config.vector_db
    # prepare_backend dispatches on vdb.type, so the narrow type holds.
    assert isinstance(vdb, MilvusConfig)
    kwargs = {"uri": vdb.resolved_uri()}
    if vdb.token:
        kwargs["token"] = vdb.token
    client = MilvusClient(**kwargs)
    try:
        if client.has_collection(vdb.collection):
            return
        dimension = resolve_embedding_dimension(config)
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)
        index = client.prepare_index_params()
        index.add_index(
            field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE"
        )
        client.create_collection(vdb.collection, schema=schema, index_params=index)
        logger.info(
            "milvus: collection %r created (dimension %d)", vdb.collection, dimension
        )
    finally:
        client.close()


def _prepare_chroma(config: ServietteConfig) -> None:
    import chromadb

    vdb = config.vector_db
    # prepare_backend dispatches on vdb.type, so the narrow type holds.
    assert isinstance(vdb, ChromaConfig)
    client = chromadb.HttpClient(
        host=vdb.host,
        port=vdb.port,
        ssl=vdb.ssl,
        headers=vdb.headers or {},
        tenant=vdb.tenant,
        database=vdb.database,
    )
    client.get_or_create_collection(vdb.collection, metadata={"hnsw:space": "cosine"})
    logger.info("chroma: collection %r ready (cosine)", vdb.collection)


def _prepare_weaviate(config: ServietteConfig) -> None:
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.connect import ConnectionParams

    vdb = config.vector_db
    # prepare_backend dispatches on vdb.type, so the narrow type holds.
    assert isinstance(vdb, WeaviateConfig)
    auth = weaviate.auth.AuthApiKey(vdb.api_key) if vdb.api_key else None
    client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host=vdb.http_host,
            http_port=vdb.http_port,
            http_secure=vdb.http_secure,
            grpc_host=vdb.grpc_host or vdb.http_host,
            grpc_port=vdb.grpc_port,
            grpc_secure=vdb.grpc_secure,
        ),
        auth_client_secret=auth,
    )
    client.connect()
    try:
        if not client.collections.exists(vdb.collection):
            client.collections.create(
                vdb.collection,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.TEXT),
                ],
            )
            logger.info("weaviate: collection %r created", vdb.collection)
    finally:
        client.close()


def _prepare_qdrant(config: ServietteConfig) -> None:
    """Create the Qdrant collection with a single named dense vector slot.

    The schema-driven ``pw.io.qdrant`` sink (Pathway ≥ 0.32.1) no longer
    auto-creates the collection: it binds each declared vector slot to the
    same-named table column, so the collection must pre-exist with a dense slot
    named ``QDRANT_VECTOR_NAME`` (matching the indexer's ``embedding`` column
    and the server accessor's search slot). The remaining columns (``text``,
    ``metadata``) land in the point payload.
    """

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    from serviette.server.accessors.qdrant import QDRANT_VECTOR_NAME

    vdb = config.vector_db
    # prepare_backend dispatches on vdb.type, so the narrow type holds.
    assert isinstance(vdb, QdrantConfig)
    client = QdrantClient(url=vdb.rest_url(), api_key=vdb.api_key)
    try:
        if not client.collection_exists(vdb.collection):
            dimension = resolve_embedding_dimension(config)
            client.create_collection(
                vdb.collection,
                vectors_config={
                    QDRANT_VECTOR_NAME: VectorParams(
                        size=dimension, distance=Distance.COSINE
                    )
                },
            )
            logger.info(
                "qdrant: collection %r created (dimension %d, vector %r)",
                vdb.collection,
                dimension,
                QDRANT_VECTOR_NAME,
            )
    finally:
        client.close()


def _prepare_pinecone(config: ServietteConfig) -> None:
    import os

    from pinecone import Pinecone, ServerlessSpec

    vdb = config.vector_db
    # prepare_backend dispatches on vdb.type, so the narrow type holds.
    assert isinstance(vdb, PineconeConfig)
    api_key = vdb.api_key or os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key, host=vdb.host)
    if not pc.has_index(vdb.index_name):
        dimension = resolve_embedding_dimension(config)
        pc.create_index(
            name=vdb.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=vdb.cloud, region=vdb.region),
        )
        logger.info(
            "pinecone: index %r created (dimension %d)", vdb.index_name, dimension
        )


def _prepare_mongodb(config: ServietteConfig) -> None:
    from pymongo import MongoClient
    from pymongo.operations import SearchIndexModel

    vdb = config.vector_db
    # prepare_backend dispatches on vdb.type, so the narrow type holds.
    assert isinstance(vdb, MongoDbConfig)
    client: MongoClient = MongoClient(vdb.connection_string)
    try:
        db = client[vdb.database]
        if vdb.collection not in db.list_collection_names():
            db.create_collection(vdb.collection)
        coll = db[vdb.collection]
        # Atlas Local boots mongot (the Search Index Management service)
        # noticeably later than mongod itself; both listing and creating
        # search indexes go through it. Ride out the boot window instead of
        # failing the first indexer start.
        import time

        from pymongo.errors import OperationFailure

        deadline = time.monotonic() + 120
        while True:
            try:
                existing = list(coll.list_search_indexes(vdb.vector_index))
                break
            except OperationFailure as exc:
                if (
                    "Search Index Management" not in str(exc)
                    or time.monotonic() > deadline
                ):
                    raise
                time.sleep(3)
        if existing:
            return
        dimension = resolve_embedding_dimension(config)
        try:
            coll.create_search_index(
                SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": dimension,
                                "similarity": "cosine",
                            }
                        ]
                    },
                    name=vdb.vector_index,
                    type="vectorSearch",
                )
            )
            logger.info(
                "mongodb: vectorSearch index %r created (dimension %d); it "
                "becomes queryable once Atlas finishes building it",
                vdb.vector_index,
                dimension,
            )
        except Exception as exc:  # noqa: BLE001 - plain MongoDB has no mongot
            logger.warning(
                "Could not create the vectorSearch index (%s). Vector search "
                "needs MongoDB Atlas or the atlas-local image; create the "
                "index manually if your deployment supports it.",
                exc,
            )
    finally:
        client.close()


def _prepare_duckdb(config: ServietteConfig) -> None:
    """Create the database file and an empty table immediately.

    The pw.io.duckdb writer would create both on its first flush, but that
    only happens once there is data: with an empty source folder the file
    would never appear and `serviette up` could not tell "still indexing" from
    "nothing to index". An empty, correctly-shaped table makes the store
    queryable from second one. The DDL mirrors the writer exactly — same
    columns, and PRIMARY KEY(chunk_id) that snapshot-mode upserts require —
    so the writer's CREATE IF NOT EXISTS + preflight accept it as-is.

    The file is pinned to storage format v1.0.0: the engine bundles its own
    DuckDB, which reads a bounded range of storage versions, while the Python
    package creating this file upgrades independently (its dev builds write
    the unreleased format 999 that no stable reader accepts). The pin keeps
    the file readable by both sides regardless of that drift.
    """

    import duckdb

    vdb = config.vector_db
    # prepare_backend dispatches on vdb.type, so the narrow type holds.
    assert isinstance(vdb, DuckDbConfig)
    Path(vdb.path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect()
    try:
        target = str(vdb.path).replace("'", "''")
        try:
            conn.execute(f"ATTACH '{target}' AS store (STORAGE_VERSION 'v1.0.0')")
        except duckdb.Error:
            # duckdb < 1.2 lacks the STORAGE_VERSION option; those builds
            # write the v1.0.0 format natively, so a plain attach is the same.
            conn.execute(f"ATTACH '{target}' AS store")
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS store."{vdb.table}" ('
            f'"chunk_id" VARCHAR PRIMARY KEY, "text" VARCHAR, '
            f'"metadata" VARCHAR, "embedding" DOUBLE[])'
        )
    finally:
        conn.close()
