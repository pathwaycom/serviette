# vetosh

**A universal, no-code RAG server for any vector database — powered by the
[Pathway](https://pathway.com) Live Data Framework.**

vetosh lets you stand up production Retrieval-Augmented Generation over your own
documents without writing code. You point it at a folder, choose a vector
database and an embedder in a YAML file (or generate one with an interactive
wizard), and run two commands:

- **`vetosh indexer`** — a Pathway streaming pipeline that watches your files,
  parses and chunks them, embeds the chunks and keeps your vector DB in sync
  (additions, modifications and deletions) in real time.
- **`vetosh server`** — a FastAPI service that embeds incoming queries and
  retrieves the most relevant chunks (and, optionally, answers them with an
  LLM). It also serves the web chat UI on `/` (same port, same origin) unless
  `server.serve_frontend` is disabled; the REST API lives under `/api/v1`.

The indexer and the server are fully decoupled: they share only the vector
database, so you can scale them independently.

---

## 1. Installation

vetosh requires **Python ≥ 3.10** (the minimum supported by Pathway).

```bash
pip install vetosh

# include the OpenAI client if you use OpenAI embedders / the /rag endpoint
pip install "vetosh[openai]"
```

For development:

```bash
pip install -e ".[dev,openai]"
```

---

## 2. Quickstart

The fastest way to get a valid config is the interactive wizard:

```bash
vetosh quickstart
```

It walks you through every option (each with a sensible default — press Enter to
accept), then writes a YAML file. You only *have* to provide document paths,
vector-DB connection details, API keys and your Pathway license key.

Then run the two components (in separate terminals or on separate machines):

```bash
# One command: indexer + server supervised together (dev/demo convenience)
vetosh up --config config.yaml         # open http://localhost:8989

# ... or run the components separately (how production deploys them):
# 1. Index your documents into the vector DB and keep it live
vetosh indexer --config config.yaml

# 2. Serve the chat UI + retrieval / RAG API on one port
vetosh server --config config.yaml     # open http://localhost:8989

# (optional, split deployments only) standalone UI tier on another host
vetosh frontend --config config.yaml   # open http://localhost:3000
```

Query the API (versioned under `/api/v1`; the pre-versioning `/retrieve`,
`/rag` and `/health` paths still work as deprecated aliases):

```bash
curl -X POST http://localhost:8989/api/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query": "how does persistence work?", "k": 5}'
```

```jsonc
{
  "results": [
    {"text": "...", "metadata": {"path": "/data/docs/guide.pdf", ...}, "score": 0.95}
  ]
}
```

If you configured an `llm` section, the `/rag` endpoint also answers questions:

```bash
curl -X POST http://localhost:8989/api/v1/rag \
  -H 'Content-Type: application/json' \
  -d '{"query": "summarize the onboarding guide", "k": 5}'
```

```jsonc
{ "answer": "...", "sources": [ {"text": "...", "metadata": {...}, "score": 0.91} ] }
```

---

## 3. Pathway License Key

Pathway requires a **free** license key. Get yours in one click at
**<https://pathway.com/framework/get-license>** (email or LinkedIn sign-in).

Provide it either via the config file:

```yaml
pathway_license_key: ${PATHWAY_LICENSE_KEY}
```

or directly as a string. Using the `${PATHWAY_LICENSE_KEY}` form keeps the key
out of the file — set it in the environment:

```bash
export PATHWAY_LICENSE_KEY="your-key-here"
```

The key is applied when the indexing graph is initialized.

---

## 4. Config reference

A single YAML file can configure both the indexer and the server (a *universal*
config), or you can split it into an indexer-only and a server-only file. Any
string value supports `${ENV_VAR}` interpolation, so credentials never need to be
hardcoded.

| Field | Type | Default | Used by | Description |
|---|---|---|---|---|
| `pathway_license_key` | str | — | indexer | Free Pathway license key (or `${ENV}`). |
| `sources` | list | — (required) | indexer | One or more sources to index (mixed types allowed). |
| `sources[].type` | `fs`\|`gdrive` | `fs` | indexer | Source type — see [Sources](#sources) below. |
| `sources[].mode` | `streaming`\|`static` | `streaming` | indexer | `streaming` watches continuously; `static` indexes once and exits. |
| `sources[].max_backlog_size` | int\|null | `1000` | indexer | **Advanced.** Backpressure bound on in-flight entries per connector (fs/gdrive/s3). Keeps memory flat during bulk backfills; the default suits virtually everyone — not offered by the wizard, edit the YAML to change. `null` disables. |
| **fs** `sources[].path` | str | — (required) | indexer | Directory to watch. |
| **fs** `sources[].glob` | str | `**/*` | indexer | Glob of files to include. |
| **gdrive** `sources[].object_id` | str | — (required) | indexer | Drive folder or file id (folders scanned recursively). |
| **gdrive** `sources[].service_user_credentials_file` | str | — (required) | indexer | Path to a Google service-account JSON credentials file. |
| **gdrive** `sources[].file_name_pattern` | str\|list | — | indexer | Optional file-name glob(s), e.g. `*.pdf`. |
| **gdrive** `sources[].object_size_limit` | int | — | indexer | Optional max object size in bytes. |
| **s3** `sources[].bucket` | str | — (required) | indexer | S3 bucket name. |
| **s3** `sources[].path` | str | `""` | indexer | Key prefix to index (`""` = whole bucket). |
| **s3** `sources[].access_key` / `.secret_access_key` | str | AWS chain | indexer | Credentials (or `${ENV}`). |
| **s3** `sources[].region` / `.endpoint` | str | — | indexer | Region; custom endpoint for MinIO etc. |
| **s3** `sources[].with_path_style` | bool | `false` | indexer | Path-style addressing (MinIO and most self-hosted). |
| **sharepoint** `sources[].url` | str | — (required) | indexer | Site URL, e.g. `https://co.sharepoint.com/sites/X`. |
| **sharepoint** `sources[].tenant` / `.client_id` | str | — (required) | indexer | App registration (certificate auth). |
| **sharepoint** `sources[].cert_path` / `.thumbprint` | str | — (required) | indexer | Certificate .pem and its thumbprint. |
| **sharepoint** `sources[].root_path` | str | — (required) | indexer | Directory/file to index. |
| **sharepoint** `sources[].recursive` | bool | `true` | indexer | Scan nested directories. |
| **sharepoint** `sources[].refresh_interval` | int | `30` | indexer | Polling period, seconds. |
| `vector_db.type` | `duckdb`\|`pgvector`\|`milvus`\|`qdrant`\|`chroma`\|`weaviate`\|`pinecone`\|`mongodb` | — (required) | both | Vector database backend (see [Backends](#vector-database-backends)). |
| **duckdb** `vector_db.path` | str | — (required) | both | Path to the DuckDB database file. |
| **duckdb** `vector_db.table` | str | `vetosh_embeddings` | both | Target table. |
| **pgvector** `vector_db.connection_string` | str | — (required) | both | `postgresql://user:pass@host/db`. |
| **pgvector** `vector_db.table` | str | `vetosh_embeddings` | both | Target table. |
| **milvus** `vector_db.uri` | str | — | both | Milvus URI, e.g. `http://localhost:19530`. |
| **milvus** `vector_db.host` / `.port` | str / int | `localhost` / `19530` | both | Used if `uri` is omitted. |
| **milvus** `vector_db.collection` | str | `vetosh_embeddings` | both | Milvus collection. |
| **qdrant** `vector_db.host` | str | `localhost` | both | Qdrant host. |
| **qdrant** `vector_db.rest_port` / `.grpc_port` | int | `6333` / `6334` | server / indexer | REST (server) and gRPC (indexer) ports. |
| **qdrant** `vector_db.api_key` | str | — | both | Qdrant Cloud API key. |
| **qdrant** `vector_db.collection` | str | `vetosh_embeddings` | both | Auto-created (cosine) if missing. |
| **chroma** `vector_db.host` / `.port` | str / int | `localhost` / `8000` | both | Chroma server address. |
| **chroma** `vector_db.tenant` / `.database` | str | Chroma defaults | both | Multi-tenant selectors. |
| **chroma** `vector_db.collection` | str | `vetosh_embeddings` | both | Must pre-exist (cosine `hnsw:space`). |
| **weaviate** `vector_db.http_host` / `.http_port` | str / int | `localhost` / `8080` | both | HTTP endpoint. |
| **weaviate** `vector_db.grpc_host` / `.grpc_port` | str / int | http_host / `50051` | server | gRPC endpoint (v4 client). |
| **weaviate** `vector_db.api_key` | str | — | both | API key. |
| **weaviate** `vector_db.collection` | str | `VetoshEmbeddings` | both | Must pre-exist (capitalized name). |
| **pinecone** `vector_db.index_name` | str | — (required) | both | Must pre-exist with the embedder's dimension. |
| **pinecone** `vector_db.api_key` | str | `${PINECONE_API_KEY}` | both | Pinecone API key. |
| **pinecone** `vector_db.namespace` | str | `""` | both | Optional namespace. |
| **mongodb** `vector_db.connection_string` | str | — (required) | both | `mongodb+srv://...` for Atlas. |
| **mongodb** `vector_db.database` / `.collection` | str | — / `vetosh_embeddings` | both | Target collection (auto-created). |
| **mongodb** `vector_db.vector_index` | str | `vector_index` | server | Atlas `vectorSearch` index name. |
| `embedder.type` | str | `openai` | both | Embedder family (see below). |
| `embedder.model` | str | provider default | both | Model name. |
| `embedder.api_key` | str | — | both | API key (or `${ENV}`). |
| `splitter.type` | `token_count`\|`recursive` | `token_count` | indexer | Chunking strategy. |
| `splitter.chunk_size` | int | `512` | indexer | Max chunk size (tokens). |
| `splitter.chunk_overlap` | int | `50` | indexer | Overlap (used by `recursive`). |
| `persistence.enabled` | bool | `true` | indexer | See [Persistence](#5-persistence). |
| `persistence.backend` | `filesystem` | `filesystem` | indexer | Persistence backend. |
| `persistence.path` | str | `./persistence` | indexer | Persistence directory (also hosts the parse cache under `runtime_calls/`). Silent default — the wizard does not ask. |
| `server.host` / `server.port` | str / int | `0.0.0.0` / `8989` | server | Bind address. |
| `server.serve_frontend` | bool | `true` | server | Serve the chat UI on `/` from the same port (API stays under `/api/v1`). |
| `llm.type` | str | `openai` | server | LLM for `/rag` (omit to disable `/rag`). |
| `llm.model` | str | `gpt-4o-mini` | server | Chat model. |
| `llm.api_key` | str | — | server | API key (or `${ENV}`). |
| `frontend.host` / `frontend.port` | str / int | `0.0.0.0` / `3000` | frontend (standalone) | Bind address for the split-deployment chat UI. |
| `frontend.api_url` | str | `http://localhost:8989` | frontend (standalone) | Base URL of the API the standalone frontend proxies to. |
| `frontend.title` | str | `vetosh` | both | Title shown in the chat UI (embedded and standalone). |

**Supported embedders** (from `pathway.xpacks.llm.embedders`): `openai`,
`litellm` (≈100 providers), `sentence_transformer` (fully local, no
credentials — install `vetosh[local]`), `gemini`, `bedrock`. Every family has a
matching async client on the server side, so one `embedder` section serves both
components; make sure the indexer and server use the **same** model so vectors
are comparable.

### Sources

vetosh only supports sources that offer Pathway's **`only_metadata`** read mode —
the graph holds just paths/identifiers and metadata (never the file bytes), and
the bytes are fetched on demand during parsing. That keeps the memory footprint
tiny (1 GB of PDFs stays a few KB in the pipeline). Today that means:

- **`fs`** — a local directory, watched recursively per `glob`.
- **`gdrive`** — a Google Drive folder or file. Create a Google **service
  account**, download its JSON key, share the target folder with the service
  account's email, and point `service_user_credentials_file` at the key. See the
  [Pathway Google Drive connector guide](https://pathway.com/developers/user-guide/connect/connectors/gdrive-connector/).
  Install the extra: `pip install "vetosh[gdrive]"`.
- **`s3`** — an S3 bucket or key prefix, including S3-compatible stores
  (MinIO, DigitalOcean, Wasabi) via `endpoint` + `with_path_style: true`.
  Credentials fall back to the standard AWS chain when omitted. No extra
  needed (boto3 ships with Pathway).
- **`sharepoint`** — a SharePoint directory or file, authenticated with an
  app-registration certificate (`tenant`, `client_id`, `cert_path`,
  `thumbprint`). Requires a Pathway **Scale** license and
  `pip install "vetosh[sharepoint]"`.

You can list several sources of mixed types. **pyfilesystem** gained
`only_metadata` support in the same Pathway change (#10483) and is next in
line to be wired into `sources:`.

### Vector database backends

Every backend is written through Pathway's **native `pw.io.*` output
connector** in snapshot/upsert mode keyed by the chunk id, so file additions,
edits and deletions become real inserts/updates/deletes in the store. The
server retrieves through each database's own vector-search API — there are no
linear scans in Python anywhere.

#### DuckDB (embedded — the zero-setup default)

```yaml
vector_db:
  type: duckdb
  path: ./embeddings.duckdb
  table: vetosh_embeddings
```

No external service and no setup: the file and table are created automatically
(`pw.io.duckdb`, snapshot mode). Embeddings are stored as native `DOUBLE[]`
columns and queried **inside DuckDB** with `list_cosine_similarity` — a
vectorized, columnar scan that answers in milliseconds for local corpora.

Concurrency: DuckDB allows one read-write process *or* several read-only
processes per file — never both. vetosh resolves this with Pathway's
`detach_between_batches`: the streaming indexer releases the file lock
between minibatches, and the server's short-lived read-only connections (with
a retry through brief lock windows) query it concurrently — live indexing and
serving work on one file. On older Pathway builds without the flag, vetosh
warns and falls back to the hold-the-lock behavior (use `mode: static`
there).

#### pgvector table

Create the target table with a `vector(n)` column sized to your embedder's
dimension, e.g. for `text-embedding-3-small` (1536 dims):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE vetosh_embeddings (
  chunk_id  text PRIMARY KEY,
  text      text,
  metadata  jsonb,
  embedding vector(1536)
);
```

(The indexer writes the chunk primary key to the `chunk_id` column.)

#### Milvus collection

Create the collection ahead of time with a primary key field `id` (VARCHAR), a
`text` (VARCHAR) field, a `metadata` (JSON) field and an `embedding`
(FLOAT_VECTOR) field of the right dimension, indexed with the `COSINE` metric.

#### Qdrant

Nothing to create: the indexer (`pw.io.qdrant`, gRPC) auto-creates the
collection with the cosine metric on first write. The chunk's `text` and
`metadata` travel in the point payload; the server queries over REST.
Set `api_key` for Qdrant Cloud.

#### ChromaDB

Create the collection first (the connector never creates it), with the cosine
distance:

```python
client.create_collection("vetosh_embeddings", metadata={"hnsw:space": "cosine"})
```

The chunk text is stored as the Chroma document and the source metadata as a
JSON string (Chroma metadata values must be scalars).

#### Weaviate

Create the collection first (capitalized name, e.g. `VetoshEmbeddings`). The
chunk vector is stored as the object vector; `text` / `metadata` (JSON string)
/ `chunk_id` become properties. The server's v4 client needs both the HTTP and
gRPC ports.

#### Pinecone

Create the index first with your embedder's dimension (cosine metric). The
`text` and JSON-string `metadata` are stored as record metadata. The API key
comes from `api_key` or `$PINECONE_API_KEY`.

#### MongoDB Atlas Vector Search

The indexer writes one document per chunk (`pw.io.mongodb`, snapshot mode);
the database/collection are auto-created. Create an Atlas `vectorSearch` index
(default name `vector_index`) on the `embedding` path with `numDimensions`
matching your embedder and the `cosine` similarity; the server retrieves with
`$vectorSearch`.

---

## 5. Persistence

When `persistence.enabled: true`, the indexer uses Pathway's built-in
persistence backend plus a `DefaultCache` UDF call-cache on the embedder. On
restart, Pathway replays its persisted state and re-emits only the *diffs* since
the last run, so:

- **unchanged documents are not re-embedded** (the embedder cache and operator
  state are restored), and
- **documents removed while the indexer was down are correctly retracted** from
  the vector DB on the next run.

The same persistence directory also hosts the **parse cache**
(`runtime_calls/`, via `pw.udfs.DefaultCache` — diskcache, LRU-bounded by
`indexer.parse_cache_size_gb`, default 8): extracted document text stays warm
across restarts, so unchanged documents are neither re-downloaded nor
re-parsed. Disabling persistence also disables the parse cache.

Disabling persistence still produces identical vectors for a given set of files;
it only forgoes the cross-restart diffing and caching.

---

## 6. Unsupported formats

vetosh never implements its own parsers — it dispatches each file to a
`pathway.xpacks.llm.parsers` parser by extension:

- `.pdf` → `PypdfParser`
- `.txt`, `.md`, `.markdown`, `.text` → `Utf8Parser`
- everything else → `UnstructuredParser` (DOCX, PPTX, XLSX, HTML, EML, CSV, RTF,
  EPUB and the other formats supported by [unstructured](https://docs.unstructured.io/))

Not supported out of the box:

- **Images** (`.png`, `.jpg`, …) and **audio/video** — these require the vision
  (`ImageParser`/`SlideParser`) or Whisper (`AudioParser`) parsers, which need a
  multimodal model and are not wired into the default dispatch.
- **Encrypted / password-protected PDFs.**
- **Google-native files** (Google Docs/Sheets/Slides) from a `gdrive` source —
  they have no direct binary download and would need an export step (TODO).
  Regular uploaded files (PDF, DOCX, …) in Drive work fine.
- Any format not handled by the `unstructured` library version you have
  installed.

Unsupported files are skipped (they simply produce no chunks).

---

## 7. Architecture

```
                 ┌──────────────────────── vetosh indexer (Pathway) ───────────────────────┐
                 │                                                                          │
   fs · gdrive   │   read(only_metadata)      parse UDF        splitter      embedder       │
   s3 · sharept  │   ┌──────────────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐    │
  ┌───────────┐──┼──▶│ paths + metadata │──▶│ extract   │──▶│  chunk   │──▶│ vectors  │──┐ │
  └───────────┘  │   │ (no file bytes)  │   │ text      │   │          │   │          │  │ │
                 │   └──────────────────┘   └─────┬─────┘   └──────────┘   └──────────┘  │ │
                 │                                │ disk parse cache (DefaultCache        │ │
                 │                                │ in <persistence>/runtime_calls,       │ │
                 │                                │ LRU-bounded)                          │ │
                 └────────────────────────────────┼──────────────────────────────────────┼─┘
                                                  │                                      │
                                                  ▼                                      ▼
                                         ┌──────────────────────────────────────────────────┐
                                         │              Vector database                      │
                                         │   (duckdb / pgvector / milvus / qdrant /          │
                                         │    chroma / weaviate / pinecone / mongodb)        │
                                         └──────────────────────────────────────────────────┘
                                                  ▲
                                                  │ top-k nearest chunks
                 ┌────────────────────────────────┼──────────────────────────────────────┐
                 │  chat UI on "/" + /api/v1/*    │                  vetosh server         │
   client  ──────┼──▶ embed query ────────────────┘  (FastAPI, async, horizontally        │
                 │                                     scalable, decoupled from indexer)   │
                 └───────────────────────────────────────────────────────────────────────┘
```

**Why `only_metadata`?** The filesystem connector only ever puts *paths and
metadata* into the graph, never file contents — so 1 GB of PDFs costs a few KB in
the pipeline. Text is extracted on demand inside the parse UDF.

**Deletion handling.** The parse UDF is non-deterministic, so Pathway memoizes
its output and, when a file is removed, re-emits that stored text *negated*
without re-reading the (now gone) file; the retraction flows through chunking and
embedding and the sink deletes exactly the matching vectors.

---

## 8. Scaling

The indexer and the server are **separate processes that share only the vector
database**:

- **Server.** Stateless and read-only against the vector DB — run as many
  instances as you like behind a load balancer, across machines or availability
  zones. Because it is fully async (coroutines, not a thread pool), each instance
  handles many concurrent network-bound retrieval calls efficiently. Each
  instance also serves the chat UI (static, zero overhead), so the UI scales
  with the API for free.
- **Frontend (optional split tier).** When the UI must live on a different
  host than the API, `vetosh frontend` is a separate, stateless web tier that
  serves the same chat page and proxies to the API at `frontend.api_url`
  server-side (the API address never reaches the browser).
- **Indexer.** Runs as its own long-lived process and can be scaled with
  Pathway's multi-worker support; its persistence lets it resume without
  re-embedding.
- **Vector database.** Scaling and replicating the vector DB itself is the
  user's responsibility and follows that database's own guidance. (The embedded
  DuckDB backend is the exception: it lives in one file next to the indexer and
  is meant for local / single-node setups.)
```
