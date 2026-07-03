"""Pathway graph construction for the indexer.

Mirrors the llm-app ``document_indexing`` template as closely as the
"external vector DB" goal allows, and reuses ``pathway.xpacks.llm`` parsers,
splitters and embedders without modification.

Pipeline
--------
``pw.io.fs.read(format="only_metadata")``  ->  parse UDF (reads bytes from the
path, extracts text via an xpack parser, memoised + persistently cached)  ->
splitter (xpack)  ->  flatten to one row per chunk  ->  embedder (xpack)  ->
vector-DB sink (duckdb / pgvector / milvus / qdrant / chroma / weaviate /
pinecone / mongodb) — every sink is a native ``pw.io`` connector writing in
snapshot/upsert mode, so retractions become real deletes in the target store.

Deletion semantics
------------------
The parse UDF is registered ``deterministic=False`` (Pathway's default), so on a
file removal Pathway re-emits the memoised parsed text *negated* and never
re-reads the now-deleted bytes; the retraction flows through split/embed and the
sink removes exactly the matching vectors.

Parse caching
-------------
Cross-restart parse caching is delegated entirely to Pathway:
``pw.udfs.DefaultCache`` stores results on disk (diskcache, LRU-bounded) under
``<persistence dir>/runtime_calls`` whenever persistence is enabled — which is
the default. No caching machinery lives in vetosh; disabling persistence also
disables the parse cache (every restart re-fetches and re-parses).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
from typing import Any

import pathway as pw

from vetosh.config.schema import VetoshConfig
from vetosh.indexer.sources import Fetcher, make_fetcher, read_source

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for working with Pathway values imperatively
# ---------------------------------------------------------------------------


def _json_to_dict(value: Any) -> dict[str, Any]:
    """Coerce a Pathway ``Json`` (or already-plain) value into a dict."""

    if isinstance(value, dict):
        return value
    # pw.Json exposes the wrapped python value via ``.value`` / ``as_dict``.
    if hasattr(value, "as_dict"):
        return dict(value.as_dict())
    if hasattr(value, "value"):
        return dict(value.value)
    return dict(value)


# ---------------------------------------------------------------------------
# Parser dispatch (xpack parsers, called imperatively because only_metadata
# means we hold paths, not bytes, in the graph)
# ---------------------------------------------------------------------------


class ParserRegistry:
    """Lazily-built, extension-dispatched xpack parsers.

    We never implement our own parsing: each file is handed to the appropriate
    ``pathway.xpacks.llm.parsers`` class. Parsers expose their logic via
    ``__wrapped__(contents)`` returning ``list[(text, metadata)]``; some are
    async, which we drive to completion here so the enclosing UDF stays sync.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def _get(self, kind: str):
        if kind not in self._cache:
            from pathway.xpacks.llm import parsers

            if kind == "pdf":
                self._cache[kind] = parsers.PypdfParser()
            elif kind == "text":
                self._cache[kind] = parsers.Utf8Parser()
            else:  # everything else -> unstructured (DOCX/HTML/PPTX/EML/...)
                self._cache[kind] = parsers.UnstructuredParser()
        return self._cache[kind]

    @staticmethod
    def _kind_for(suffix: str) -> str:
        suffix = suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        if suffix in {".txt", ".md", ".markdown", ".text", ""}:
            return "text"
        return "unstructured"

    def parse(self, contents: bytes, suffix: str) -> str:
        parser = self._get(self._kind_for(suffix))
        result = parser.__wrapped__(contents)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        # result is list[(text, metadata)]; concatenate element texts — our own
        # splitter re-chunks downstream.
        return "\n\n".join(text for text, _meta in result if text)


# ---------------------------------------------------------------------------
# Embedder / splitter builders (xpack)
# ---------------------------------------------------------------------------


def build_xpack_embedder(cfg) -> pw.UDF:
    """Construct a ``pathway.xpacks.llm.embedders`` UDF from config.

    A ``DefaultCache`` strategy is attached so identical chunks are not
    re-embedded across runs (it uses the persistence layer when enabled).
    """

    if cfg.type == "mock":
        # Test-only deterministic embedder; runs without any provider/credentials.
        from vetosh.testing import build_mock_embedder

        return build_mock_embedder()

    from pathway.xpacks.llm import embedders

    cache_strategy = pw.udfs.DefaultCache()
    extra = {
        k: v
        for k, v in cfg.model_dump(exclude={"type", "model", "api_key"}).items()
        if v is not None
    }
    common: dict[str, Any] = {"cache_strategy": cache_strategy, **extra}
    if cfg.model:
        common["model"] = cfg.model

    if cfg.type == "openai":
        return embedders.OpenAIEmbedder(api_key=cfg.api_key, **common)
    if cfg.type == "litellm":
        return embedders.LiteLLMEmbedder(api_key=cfg.api_key, **common)
    if cfg.type in {"sentence_transformer", "sentencetransformer"}:
        model = common.pop("model", None) or "sentence-transformers/all-MiniLM-L6-v2"
        common.pop("cache_strategy", None)  # local model; caching adds little
        udf = embedders.SentenceTransformerEmbedder(model=model, **common)
        # Local models are deterministic: the engine may re-run them on
        # retraction (cheap CPU) instead of memoizing every vector in RAM.
        # The xpack constructor doesn't expose the flag (upstream PR pending),
        # so it is set on the built UDF; the engine reads it at expression
        # build time. API embedders stay memoized — a re-call costs money and
        # is not guaranteed bit-stable. Model changes across restarts are
        # guarded by the persistence fingerprint.
        udf.deterministic = True
        return udf
    if cfg.type == "gemini":
        return embedders.GeminiEmbedder(api_key=cfg.api_key, **common)
    if cfg.type == "bedrock":
        return embedders.BedrockEmbedder(**common)
    raise ValueError(f"Unsupported embedder type: {cfg.type!r}")


def build_xpack_splitter(cfg):
    """Construct an xpack splitter. ``token_count`` is the default."""

    from pathway.xpacks.llm import splitters

    if cfg.type in {"token_count", "tokencount"}:
        # TokenCountSplitter is token-based; map chunk_size -> max_tokens.
        return splitters.TokenCountSplitter(max_tokens=cfg.chunk_size)
    if cfg.type in {"recursive", "recursive_character"}:
        return splitters.RecursiveSplitter(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )
    if cfg.type in {"null", "none"}:
        return splitters.NullSplitter()
    raise ValueError(f"Unsupported splitter type: {cfg.type!r}")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(
    config: VetoshConfig,
    *,
    embedder: pw.UDF | None = None,
    splitter=None,
) -> pw.Table:
    """Build the indexing graph and return the final embeddings table.

    ``embedder``/``splitter`` can be injected (tests pass a mock embedder);
    otherwise they are built from ``config``.
    """

    config.for_indexer()

    registry = ParserRegistry()
    splitter = splitter if splitter is not None else build_xpack_splitter(config.splitter)
    embedder = embedder if embedder is not None else build_xpack_embedder(config.embedder)

    # -- parse UDF factory ---------------------------------------------------
    # deterministic=False => memoised; never re-runs on retraction (the source
    # object is gone). DefaultCache persists parsed text across restarts on
    # disk when persistence is enabled (LRU-bounded; size from config). The
    # fetcher makes byte retrieval source-specific (local path vs Drive download).
    cache_strategy = pw.udfs.DefaultCache(
        size_limit=config.indexer.parse_cache_size_gb * 2**30
    )

    def make_parse_udf(fetcher: Fetcher):
        @pw.udf(deterministic=False, cache_strategy=cache_strategy)
        def parse_document(metadata: pw.Json) -> str:
            meta = _json_to_dict(metadata)
            try:
                contents, suffix = fetcher.fetch(meta)
            except Exception as exc:  # noqa: BLE001 - object may have vanished / be unreadable
                logger.warning("Could not fetch source object %s: %s", meta, exc)
                return ""
            return registry.parse(contents, suffix)

        return parse_document

    # Pure-function whitelist: for these splitter types re-running the UDF is
    # guaranteed to reproduce the original chunks, so the engine need not
    # memoize its outputs (a full extra copy of the corpus in RAM). Guarded
    # against config/library drift by the persistence fingerprint (see
    # fingerprint.py). Unknown/future splitter types fall back to memoization.
    _PURE_SPLITTERS = {"token_count", "tokencount", "recursive", "recursive_character", "null", "none"}
    split_is_pure = config.splitter.type in _PURE_SPLITTERS

    @pw.udf(deterministic=split_is_pure)
    def split_text(text: str) -> list[str]:
        if not text:
            return []
        return [chunk for chunk, _meta in splitter.chunk(text)]

    @pw.udf(deterministic=True)
    def make_id(meta_json: str, text: str) -> str:
        # meta_json is the canonical per-document serialization from
        # _metadata_as_json, so the id is reproducible for (metadata, text).
        digest = hashlib.sha256()
        digest.update(meta_json.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    # -- per-source: read (only_metadata) -> parse ----------------------------
    # Each source gets its own fetcher-bound parse UDF; parsed tables share the
    # (_metadata, text) schema and are concatenated before splitting/embedding.
    parsed_tables: list[pw.Table] = []
    for i, src in enumerate(config.sources):
        table = read_source(src, name=f"source_{i}")
        parse_document = make_parse_udf(make_fetcher(src))
        parsed_tables.append(
            table.select(_metadata=pw.this._metadata, text=parse_document(pw.this._metadata))
        )

    parsed = (
        parsed_tables[0]
        if len(parsed_tables) == 1
        else pw.Table.concat_reindex(*parsed_tables)
    )

    # -- split -> flatten -> embed -------------------------------------------
    # The canonical metadata JSON is computed once per DOCUMENT here; flatten
    # then replicates the reference per chunk instead of re-serializing the
    # same dict for every chunk.
    chunked = parsed.select(
        _metadata=pw.this._metadata,
        meta_json=_metadata_as_json(pw.this._metadata),
        chunk=split_text(pw.this.text),
    )
    exploded = chunked.flatten(pw.this.chunk)
    # Pathway reserves the column name "id", so the chunk's primary key lives in
    # "chunk_id"; the sinks map it to each backend's id/primary-key field.
    embedded = exploded.select(
        chunk_id=make_id(pw.this.meta_json, pw.this.chunk),
        text=pw.this.chunk,
        metadata=pw.this._metadata,
        metadata_json=pw.this.meta_json,
        embedding=embedder(pw.this.chunk),
    )

    _write_sink(embedded, config)
    return embedded


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


def _write_sink(table: pw.Table, config: VetoshConfig) -> None:
    """Route the embedded table to the configured backend sink.

    ``table`` carries the metadata twice (native ``metadata`` Json and the
    canonical ``metadata_json`` string); each sink selects exactly the columns
    its backend stores, so nothing is written twice and backends that key rows
    internally (qdrant, mongodb) don't carry a dead ``chunk_id`` field.
    """

    vdb = config.vector_db
    # Backends storing native JSON metadata, keyed by chunk_id.
    plain = table.select(
        chunk_id=pw.this.chunk_id,
        text=pw.this.text,
        metadata=pw.this.metadata,
        embedding=pw.this.embedding,
    )
    # Backends restricted to scalar record metadata: the JSON string form.
    jsonified = table.select(
        chunk_id=pw.this.chunk_id,
        text=pw.this.text,
        metadata=pw.this.metadata_json,
        embedding=pw.this.embedding,
    )
    if vdb.type == "duckdb":
        _write_duckdb(jsonified, vdb)
    elif vdb.type == "pgvector":
        _write_pgvector(plain, vdb)
    elif vdb.type == "milvus":
        _write_milvus(plain, vdb)
    elif vdb.type == "qdrant":
        # No chunk_id: the sink keys points internally; payload = text + metadata.
        _write_qdrant(plain.without(pw.this.chunk_id), vdb)
    elif vdb.type == "chroma":
        _write_chroma(jsonified, vdb)
    elif vdb.type == "weaviate":
        _write_weaviate(jsonified, vdb)
    elif vdb.type == "pinecone":
        _write_pinecone(jsonified, vdb)
    elif vdb.type == "mongodb":
        # No chunk_id: snapshot mode keys documents by the internal _id.
        _write_mongodb(plain.without(pw.this.chunk_id), vdb)
    else:  # pragma: no cover - guarded by schema
        raise ValueError(f"Unsupported vector_db type: {vdb.type!r}")


@pw.udf(deterministic=True)
def _metadata_as_json(metadata: pw.Json) -> str:
    """Serialize the metadata dict to one canonical JSON string.

    Computed once per *document* (before chunk flattening) and reused for
    every chunk: as the string form stored by backends whose record metadata
    must be scalar (duckdb/chroma/weaviate/pinecone), and as the metadata part
    of the chunk id hash. sort_keys/ensure_ascii keep it byte-stable.
    """

    return json.dumps(_json_to_dict(metadata), sort_keys=True, ensure_ascii=True)


def _write_pgvector(table: pw.Table, vdb) -> None:
    """Write to Postgres/pgvector in snapshot mode so retractions delete rows.

    The target table must exist with an ``embedding vector(n)`` column (see
    docs/README "Config reference"). ``output_table_type='snapshot'`` keeps the
    table as an exact replica of the current chunk set, issuing real
    INSERT/UPDATE/DELETE keyed by ``id``.
    """

    settings = _libpq_settings(vdb.connection_string)
    pw.io.postgres.write(
        table,
        settings,
        vdb.table,
        output_table_type="snapshot",
        primary_key=[table.chunk_id],
    )


def _write_milvus(table: pw.Table, vdb) -> None:
    pw.io.milvus.write(
        table,
        uri=vdb.resolved_uri(),
        collection_name=vdb.collection,
        primary_key=table.chunk_id,
    )


def _write_duckdb(table: pw.Table, vdb) -> None:
    """Write to an embedded DuckDB file via Pathway's native connector.

    Snapshot mode keyed by ``chunk_id`` keeps the table an exact replica of the
    current chunk set (real upserts/deletes). Embeddings land as native
    ``DOUBLE[]`` lists that the server queries in-database with
    ``list_cosine_similarity``. The metadata travels as a JSON string: DuckDB
    compares identifiers case-insensitively and stores ``pw.Json`` as text
    anyway, and one explicit column keeps the accessor symmetric with the
    other backends.
    """

    out = table
    kwargs: dict[str, Any] = {}
    # Release the single-writer file lock between minibatches so a separate
    # server process can answer queries while the streaming indexer runs
    # (pathway >= #10496; the accessor retries through the brief lock
    # windows). Older builds keep the previous hold-the-lock behavior.
    if "detach_between_batches" in inspect.signature(pw.io.duckdb.write).parameters:
        kwargs["detach_between_batches"] = True
    else:
        logger.warning(
            "This pathway build lacks duckdb detach_between_batches: while a "
            "streaming indexer runs, a separate server process cannot read "
            "the database file."
        )
    pw.io.duckdb.write(
        out,
        table_name=vdb.table,
        database=vdb.path,
        output_table_type="snapshot",
        primary_key=[out.chunk_id],
        init_mode="create_if_not_exists",
        **kwargs,
    )


def _write_qdrant(table: pw.Table, vdb) -> None:
    """Write points to Qdrant (auto-creates a cosine collection if missing).

    The sink keys points internally per row, so additions/updates/deletions
    map to native upserts/deletes; the remaining columns (``chunk_id``,
    ``text``, ``metadata``) become the point payload.
    """

    pw.io.qdrant.write(
        table,
        vdb.grpc_url(),
        vdb.collection,
        vector=table.embedding,
        api_key=vdb.api_key,
    )


def _write_chroma(table: pw.Table, vdb) -> None:
    """Write to a pre-existing ChromaDB collection (cosine ``hnsw:space``)."""

    out = table
    pw.io.chroma.write(
        out,
        vdb.collection,
        primary_key=out.chunk_id,
        embedding=out.embedding,
        document=out.text,
        metadata_columns=[out.metadata],
        host=vdb.host,
        port=vdb.port,
        ssl=vdb.ssl,
        headers=vdb.headers,
        tenant=vdb.tenant,
        database=vdb.database,
    )


def _write_weaviate(table: pw.Table, vdb) -> None:
    """Write to a pre-existing Weaviate collection (object vector + properties)."""

    out = table
    pw.io.weaviate.write(
        out,
        vdb.collection,
        primary_key=out.chunk_id,
        vector=out.embedding,
        http_host=vdb.http_host,
        http_port=vdb.http_port,
        http_secure=vdb.http_secure,
        api_key=vdb.api_key,
    )


def _write_pinecone(table: pw.Table, vdb) -> None:
    """Write to a pre-existing Pinecone index (dimension must match)."""

    out = table
    pw.io.pinecone.write(
        out,
        vdb.index_name,
        primary_key=out.chunk_id,
        vector=out.embedding,
        api_key=vdb.api_key,
        host=vdb.host,
        namespace=vdb.namespace,
        metadata_columns=[out.text, out.metadata],
    )


def _write_mongodb(table: pw.Table, vdb) -> None:
    """Write documents to MongoDB / Atlas in snapshot mode.

    One document per chunk with the embedding as a BSON number array — exactly
    the shape Atlas Vector Search queries with ``$vectorSearch`` (the user
    creates the ``vectorSearch`` index; Pathway cannot know the dimension).
    """

    pw.io.mongodb.write(
        table,
        connection_string=vdb.connection_string,
        database=vdb.database,
        collection=vdb.collection,
        output_table_type="snapshot",
    )


def _libpq_settings(connection_string: str) -> dict[str, Any]:
    """Parse a ``postgresql://`` URL into a pw.io.postgres settings dict."""

    from urllib.parse import unquote, urlparse

    parsed = urlparse(connection_string)
    settings: dict[str, Any] = {}
    if parsed.hostname:
        settings["host"] = parsed.hostname
    if parsed.port:
        settings["port"] = parsed.port
    if parsed.username:
        settings["user"] = unquote(parsed.username)
    if parsed.password:
        settings["password"] = unquote(parsed.password)
    dbname = parsed.path.lstrip("/")
    if dbname:
        settings["dbname"] = dbname
    return settings


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def persistence_config(config: VetoshConfig):
    """Build a ``pw.persistence.Config`` (or None) from config.

    Enabling persistence prevents re-embedding unchanged documents on restart:
    Pathway replays operator state and the embedder's ``DefaultCache`` is backed
    by this layer.
    """

    if not config.persistence.enabled:
        return None
    backend = pw.persistence.Backend.filesystem(config.persistence.path)
    return pw.persistence.Config(backend)


def run_indexer(
    config: VetoshConfig, *, prepare: bool = True, **build_kwargs: Any
) -> None:
    """Prepare the backend, build the graph and run it (streaming).

    ``prepare=False`` is used by spawned worker processes: the spawn parent
    has already created the target (see ``vetosh.indexer.main``).
    """

    if config.pathway_license_key:
        pw.set_license_key(config.pathway_license_key)
    config.for_indexer()
    if prepare:
        # Refuse to run against persisted state built with an incompatible
        # config (chunking drift corrupts retractions — see fingerprint.py).
        from vetosh.indexer.fingerprint import check_fingerprint

        check_fingerprint(config)
        # Create the target table/collection/index if missing — the connectors
        # auto-create only for duckdb and qdrant (see prepare.py).
        from vetosh.indexer.prepare import prepare_backend

        prepare_backend(config)
    build_graph(config, **build_kwargs)
    pw.run(persistence_config=persistence_config(config))
