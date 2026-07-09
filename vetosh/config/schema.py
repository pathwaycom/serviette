"""Shared configuration schema for vetosh (Pydantic v2).

A single YAML file can hold both indexer and server configuration (a *universal*
config), or the user can split it into an indexer-only and a server-only file.
Most top-level sections are therefore optional at the schema level; each
component validates the subset it needs via :func:`require` at startup.

String values support ``${ENV_VAR}`` interpolation, resolved while the YAML is
loaded (see :func:`load_config`). This keeps credentials out of the config file
itself, satisfying the "no hardcoded credentials" principle.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vetosh import APP_NAME

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class FsSource(BaseModel):
    """A filesystem source watched by the indexer.

    ``type`` is the discriminator across source types. Only sources with an
    ``only_metadata`` read mode are supported (see ``GDriveSource``).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["fs"] = "fs"
    path: str
    glob: str = "**/*"
    # "streaming" watches the path continuously (default); "static" reads the
    # current contents once and lets the graph terminate (used for one-shot
    # indexing and tests).
    mode: Literal["streaming", "static"] = "streaming"
    # Bound on entries the connector keeps in flight before downstream
    # processing catches up. Backpressure keeps the indexer's memory flat
    # during bulk backfills and makes sink commits arrive steadily instead of
    # in one giant batch. None disables the bound.
    max_backlog_size: int | None = 1000


class GDriveSource(BaseModel):
    """A Google Drive source (folder or file) read in ``only_metadata`` mode.

    Like the filesystem source, the graph holds only metadata; file bytes are
    downloaded on demand by id during parsing. Requires a Google service-account
    credentials file (see the Google Drive connector docs).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["gdrive"] = "gdrive"
    object_id: str
    service_user_credentials_file: str
    # Glob (or list of globs) on file names, e.g. "*.pdf". None = no filtering.
    file_name_pattern: str | list[str] | None = None
    object_size_limit: int | None = None
    mode: Literal["streaming", "static"] = "streaming"
    # See FsSource.max_backlog_size.
    max_backlog_size: int | None = 1000


class S3Source(BaseModel):
    """An S3 bucket (or any S3-compatible store: MinIO, DigitalOcean, Wasabi)
    read in ``only_metadata`` mode: the graph tracks object metadata only, and
    object bytes are downloaded on demand during parsing. Credentials fall
    back to the standard AWS chain when omitted.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["s3"] = "s3"
    bucket: str
    # Key or key prefix to index ("" = the whole bucket).
    path: str = ""
    access_key: str | None = None
    secret_access_key: str | None = None
    region: str | None = None
    # Custom endpoint for self-hosted / S3-compatible storage (e.g. MinIO);
    # such setups usually also need path-style addressing.
    endpoint: str | None = None
    with_path_style: bool = False
    mode: Literal["streaming", "static"] = "streaming"
    # See FsSource.max_backlog_size.
    max_backlog_size: int | None = 1000

    @model_validator(mode="after")
    def _custom_endpoint_needs_region(self) -> "S3Source":
        # Without a region the AWS signature's credential scope is malformed
        # and S3-compatible stores reject requests with an opaque
        # AuthorizationQueryParametersError. MinIO's default is us-east-1.
        if self.endpoint and not self.region:
            raise ValueError(
                "s3 source: 'region' is required when 'endpoint' is set "
                "(use 'us-east-1' for a default MinIO installation)"
            )
        return self


class SharePointSource(BaseModel):
    """A Microsoft SharePoint directory/file read in ``only_metadata`` mode.

    Uses certificate authentication (an app registration with a certificate;
    see the Pathway SharePoint connector docs). Requires a Pathway Scale
    license and the ``vetosh[sharepoint]`` extra.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["sharepoint"] = "sharepoint"
    url: str  # e.g. https://company.sharepoint.com/sites/MySite
    tenant: str
    client_id: str
    cert_path: str
    thumbprint: str
    root_path: str
    recursive: bool = True
    object_size_limit: int | None = None
    refresh_interval: int = 30
    mode: Literal["streaming", "static"] = "streaming"


class PyFilesystemSource(BaseModel):
    """Any filesystem PyFilesystem can open, watched in ``only_metadata``
    mode: FTP, SFTP/SSH, WebDAV, ZIP/TAR archives, in-memory and more —
    the ``fs_url`` (e.g. ``"ftp://user:pass@host/dir"``, ``"zip://docs.zip"``)
    selects the driver. Requires the ``vetosh[pyfilesystem]`` extra; some
    protocols need their own driver package (``fs.sshfs``, ``fs.webdavfs``).
    Credentials belong in the URL via ``${ENV_VAR}`` interpolation."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pyfilesystem"] = "pyfilesystem"
    fs_url: str
    # Path inside the opened filesystem ("" = its root), scanned recursively.
    path: str = ""
    mode: Literal["streaming", "static"] = "streaming"
    # Seconds between scans in streaming mode.
    refresh_interval: float = 30.0
    # See FsSource.max_backlog_size.
    max_backlog_size: int | None = 1000


class ParserRule(BaseModel):
    """One routing rule of the ``parser:`` section: files whose names match
    any of the ``match`` globs are parsed by ``type``. Rules are checked in
    order, first match wins; built-in keyless-first defaults cover whatever
    the user rules don't. ``options`` are forwarded to the parser constructor
    (e.g. ``prompt`` for twelvelabs_video, ``model``/``api_key`` for
    whisper); ``${ENV_VAR}`` interpolation applies as everywhere."""

    model_config = ConfigDict(extra="forbid")

    match: list[str] = Field(min_length=1)
    type: Literal[
        "utf8",
        "pypdf",
        "docling",
        "unstructured",
        "paddle_ocr",
        "vision_image",
        "vision_slide",
        "whisper",
        "twelvelabs_video",
        "skip",
    ]
    options: dict[str, Any] = Field(default_factory=dict)


Source = Annotated[
    Union[FsSource, GDriveSource, S3Source, SharePointSource, PyFilesystemSource],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Vector databases
# ---------------------------------------------------------------------------


class PgVectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["pgvector"] = "pgvector"
    connection_string: str
    table: str = "vetosh_embeddings"
    # Dimension of the embedding column. Used when the indexer auto-creates the
    # table. If omitted, it is inferred from the embedder at indexer startup.
    embedding_dimension: int | None = None


class MilvusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["milvus"] = "milvus"
    # Either a full URI (e.g. "http://localhost:19530" or "./milvus.db" for
    # Milvus Lite) or host + port. ``uri`` takes precedence when provided.
    uri: str | None = None
    host: str = "localhost"
    port: int = 19530
    collection: str = "vetosh_embeddings"
    token: str | None = None
    embedding_dimension: int | None = None

    def resolved_uri(self) -> str:
        if self.uri:
            return self.uri
        return f"http://{self.host}:{self.port}"


class DuckDbConfig(BaseModel):
    """Embedded vector store backed by a local DuckDB database file.

    Zero-setup: no external service. The indexer writes through Pathway's native
    ``pw.io.duckdb`` connector in *snapshot* mode (real upserts/deletes keyed by
    chunk id); embeddings land as native ``DOUBLE[]`` lists and retrieval runs
    **inside DuckDB** with ``list_cosine_similarity`` — a vectorized, columnar
    scan, not a Python loop.

    Concurrency: the sink writes with ``detach_between_batches`` (Pathway ≥
    #10496), releasing the single-writer file lock between minibatches, so the
    server's retrying read-only connections query the same file while a
    streaming indexer runs. On older Pathway builds vetosh warns and falls
    back to hold-the-lock behavior (use ``mode: static`` there).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["duckdb"] = "duckdb"
    path: str
    table: str = "vetosh_embeddings"


class QdrantConfig(BaseModel):
    """Qdrant backend. The indexer writes via ``pw.io.qdrant`` (gRPC); the
    server queries via the REST/HTTP API. The collection is auto-created on
    first write (cosine metric) if it does not exist."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["qdrant"] = "qdrant"
    host: str = "localhost"
    rest_port: int = 6333
    grpc_port: int = 6334
    https: bool = False
    api_key: str | None = None
    collection: str = "vetosh_embeddings"

    def rest_url(self) -> str:
        scheme = "https" if self.https else "http"
        return f"{scheme}://{self.host}:{self.rest_port}"

    def grpc_url(self) -> str:
        scheme = "https" if self.https else "http"
        return f"{scheme}://{self.host}:{self.grpc_port}"


class ChromaConfig(BaseModel):
    """ChromaDB backend (HTTP client/server mode). The **collection must
    already exist** (created with the cosine ``hnsw:space``); the connector
    never creates it."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["chroma"] = "chroma"
    host: str = "localhost"
    port: int = 8000
    ssl: bool = False
    headers: dict[str, str] | None = None
    tenant: str = "default_tenant"
    database: str = "default_database"
    collection: str = "vetosh_embeddings"


class WeaviateConfig(BaseModel):
    """Weaviate backend. The **collection must already exist**. The indexer
    writes over HTTP; the server's v4 client also needs the gRPC port."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["weaviate"] = "weaviate"
    http_host: str = "localhost"
    http_port: int = 8080
    http_secure: bool = False
    grpc_host: str | None = None  # defaults to http_host
    grpc_port: int = 50051
    grpc_secure: bool = False
    api_key: str | None = None
    collection: str = "VetoshEmbeddings"


class PineconeConfig(BaseModel):
    """Pinecone backend. The **index must already exist** with a dimension
    matching the embedder (cosine metric recommended). ``host`` is only needed
    for Pinecone Local; the managed service resolves it from the API key."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pinecone"] = "pinecone"
    index_name: str
    api_key: str | None = None  # falls back to PINECONE_API_KEY
    host: str | None = None
    namespace: str = ""
    # Used when the indexer auto-creates a missing index.
    embedding_dimension: int | None = None
    cloud: str = "aws"
    region: str = "us-east-1"


class MongoDbConfig(BaseModel):
    """MongoDB Atlas Vector Search backend. The indexer writes documents in
    snapshot mode via ``pw.io.mongodb``; create an Atlas ``vectorSearch`` index
    (named ``vector_index`` by default) on the ``embedding`` path with
    ``numDimensions`` matching the embedder."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["mongodb"] = "mongodb"
    connection_string: str
    database: str
    collection: str = "vetosh_embeddings"
    vector_index: str = "vector_index"
    # Used when the indexer auto-creates a missing vectorSearch index.
    embedding_dimension: int | None = None


VectorDBConfig = Annotated[
    Union[
        DuckDbConfig,
        PgVectorConfig,
        MilvusConfig,
        QdrantConfig,
        ChromaConfig,
        WeaviateConfig,
        PineconeConfig,
        MongoDbConfig,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Embedder / splitter / LLM
# ---------------------------------------------------------------------------


class EmbedderConfig(BaseModel):
    """Embedder configuration.

    ``type`` selects an embedder family that maps onto a
    ``pathway.xpacks.llm.embedders`` class on the indexer side and onto an async
    client on the server side. Any extra keys are forwarded to the underlying
    embedder constructor / client, so provider-specific options need no schema
    change.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "openai"
    model: str | None = None
    api_key: str | None = None


class SplitterConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "token_count"
    chunk_size: int = 512
    chunk_overlap: int = 50


class LLMConfig(BaseModel):
    """Optional LLM configuration for the ``/rag`` endpoint."""

    model_config = ConfigDict(extra="allow")

    type: str = "openai"
    model: str | None = None
    api_key: str | None = None


# ---------------------------------------------------------------------------
# Persistence / server
# ---------------------------------------------------------------------------


class IndexerConfig(BaseModel):
    """Indexer runtime options."""

    model_config = ConfigDict(extra="forbid")

    # Number of Pathway worker PROCESSES. When > 1, `vetosh indexer` re-executes
    # itself through `pathway spawn --processes N` (one worker thread each);
    # sinks that key rows internally (e.g. qdrant) write in parallel.
    workers: int = Field(default=1, ge=1)
    # Advanced. Disk budget for the parse cache (pw.udfs.DefaultCache, stored
    # under <persistence dir>/runtime_calls, LRU-evicted). Size it at least to
    # the extracted-text volume of the corpus to avoid re-parse churn.
    parse_cache_size_gb: int = Field(default=8, ge=1)
    # Advanced. If set, the engine keeps the memoization cache of
    # non-deterministic UDFs (parsed texts) in files in this directory
    # instead of RAM (requires a Pathway build with pw.run's
    # udf_cache_directory). Point it at a real disk, not tmpfs.
    udf_cache_directory: str | None = None
    # Advanced. Base port of the Pathway engine's built-in monitoring HTTP
    # server: every worker process serves GET /status (JSON snapshot) and
    # GET /metrics (Prometheus) on 127.0.0.1:(port + worker index).
    # None disables the server.
    monitoring_http_port: int | None = Field(default=None, ge=1, le=65535)


class PersistenceConfig(BaseModel):
    """On-disk persistence — the silent default (the wizard does not ask).

    Carries three things: incremental state across restarts (no re-embedding
    of unchanged documents), correct retraction of files deleted while the
    indexer was down, and the parse cache (``runtime_calls``). Advanced users
    tune or disable it by editing the config directly; disabling also
    disables the parse cache.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    backend: Literal["filesystem"] = "filesystem"
    # Relative to the indexer's working directory; writable out of the box.
    path: str = "./persistence"


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Loopback by default: the server has no built-in auth (that is the
    # perimeter's job — see the Security section of the docs), so listening
    # on all interfaces must be an explicit decision ("0.0.0.0"), not a
    # surprise on a laptop in a cafe.
    host: str = "127.0.0.1"
    # An uncommon default port: 8000/8080 collide with half the dev tools.
    port: int = 8989
    # Opt-in CORS allowlist for third-party browser frontends that call this
    # API directly from another origin. Empty (the default) = no CORS
    # headers at all — which is itself a protection: it stops web pages from
    # reading responses of localhost instances (drive-by exfiltration).
    # vetosh's own UIs never need this (same origin / server-side proxy).
    # Enable only if you know what you are doing; prefer exact origins over "*".
    cors_origins: list[str] = Field(default_factory=list)
    # Serve the chat UI on "/" from the same process/port (same-origin, no
    # CORS). Disable for a pure-API deployment; the standalone
    # `vetosh frontend` command covers split UI/API deployments.
    serve_frontend: bool = True


class FrontendConfig(BaseModel):
    """Web chat UI. Decoupled from the API: it only needs the API's address.

    The frontend serves the page and *proxies* requests to ``api_url`` server-side
    (no CORS, the API address never reaches the browser), so indexing, API and
    frontend can each run on a different worker/host.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"  # loopback by default, same reasoning as ServerConfig
    port: int = 3000
    # Base URL of the vetosh server API this frontend talks to.
    api_url: str = "http://localhost:8989"
    title: str = APP_NAME


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class VetoshConfig(BaseModel):
    """Top-level universal configuration.

    Sections are optional so the same model validates indexer-only and
    server-only files. Use :meth:`for_indexer` / :meth:`for_server` to assert the
    fields a given component requires.
    """

    model_config = ConfigDict(extra="forbid")

    pathway_license_key: str | None = None

    sources: list[Source] = Field(default_factory=list)
    vector_db: VectorDBConfig | None = None
    embedder: EmbedderConfig | None = None
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    # Optional parser routing rules (see ParserRule). None: built-in defaults
    # cover every modality, preferring keyless local parsers (best available:
    # docling > pypdf for PDF, PaddleOCR for images) and skipping modalities
    # whose only parser needs an absent API key (audio, video) with a warning.
    parser: list[ParserRule] | None = None
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)

    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig | None = None
    frontend: FrontendConfig | None = None

    # -- per-component requirements -------------------------------------------

    def require(self, *fields: str) -> None:
        missing = [f for f in fields if not getattr(self, f)]
        if missing:
            raise ValueError(
                "Missing required config section(s) for this command: "
                + ", ".join(missing)
            )

    def for_indexer(self) -> VetoshConfig:
        self.require("sources", "vector_db", "embedder")
        return self

    def for_server(self) -> VetoshConfig:
        self.require("vector_db", "embedder")
        return self

    def for_frontend(self) -> VetoshConfig:
        self.require("frontend")
        return self


# ---------------------------------------------------------------------------
# Loading + ${ENV_VAR} interpolation
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def interpolate_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` occurrences in strings with env values.

    A missing variable resolves to an empty string and a warning-worthy state
    that surfaces later as a validation/connection error, rather than silently
    embedding the literal ``${VAR}`` text.
    """

    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v) for v in value]
    return value


def load_config(path: str | Path) -> VetoshConfig:
    """Load, env-interpolate and validate a YAML config file."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    interpolated = interpolate_env(raw)
    try:
        return VetoshConfig.model_validate(interpolated)
    except ValidationError as exc:  # pragma: no cover - re-raised with context
        raise ValueError(f"Invalid configuration in {path}:\n{exc}") from exc


def load_config_dict(data: dict[str, Any]) -> VetoshConfig:
    """Validate an already-parsed config dict (used by the quickstart wizard)."""

    return VetoshConfig.model_validate(interpolate_env(data))
