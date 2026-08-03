# serviette

**A universal, no-code, always up-to-date RAG server for any vector database
— powered by the [Pathway](https://pathway.com) Live Data Framework.**

serviette stands up Retrieval-Augmented Generation over your own documents —
text, Office files, PDFs, scans, and (with API keys) audio and video —
without writing code. Point it at a folder, choose a vector database and an
embedder in a YAML file (or generate one with the wizard), and run one
command (`serviette up`) or the two components separately:

- **`serviette indexer`** — a Pathway streaming pipeline that watches your files,
  parses and chunks them, embeds the chunks and keeps your vector DB in sync
  (additions, modifications and deletions) in real time.
- **`serviette server`** — a FastAPI service that embeds incoming queries and
  retrieves the most relevant chunks (and, optionally, answers them with an
  LLM). It also serves the web chat UI on `/` (same port, same origin) unless
  `server.serve_frontend` is disabled; the REST API lives under `/api/v1`.

The indexer and the server are fully decoupled: they share only the vector
database, so you can scale them independently.

---

## 1. Installation

serviette requires **Python ≥ 3.10** (the minimum supported by Pathway).

```bash
pip install serviette
```

(or, from a checkout, `pip install -e ".[dev,local]"` — see the README's
[Development](../README.md#development) section for the full walkthrough).
OpenAI support (the default `/rag` LLM and the OpenAI embedder) is built
in — just set `OPENAI_API_KEY`.

Optional extras: `local` (sentence-transformers — free local embeddings),
`docling` (layout-aware PDF + Office parsing), `ocr` (scanned images),
`pyfilesystem` (FTP/SFTP/WebDAV/ZIP sources), `gdrive`, `sharepoint`, plus
one extra per vector-DB client (`qdrant`, `pgvector`, …, or `all`).

---

## 2. Quickstart

The fastest way to get a valid config is the interactive wizard:

```bash
serviette quickstart
```

It asks only what it must (the DuckDB + local-embeddings happy path is seven
questions) and writes a YAML file; everything else — port, persistence,
chunking, parser routing — gets silent, documented defaults you can edit
later. You only *have* to provide document paths, vector-DB connection
details (for client-server backends), API keys for keyed providers, and your
Pathway license key.

Then run the two components (in separate terminals or on separate machines):

```bash
# One command: indexer + server supervised together (dev/demo convenience)
serviette up --config config.yaml         # open http://localhost:8989

# ... or run the components separately (how production deploys them):
# 1. Index your documents into the vector DB and keep it live
serviette indexer --config config.yaml

# 2. Serve the chat UI + retrieval / RAG API on one port
serviette server --config config.yaml     # open http://localhost:8989

# (optional, split deployments only) standalone UI tier on another host
serviette frontend --config config.yaml   # open http://localhost:3000
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
| `sources[].type` | `fs`\|`gdrive`\|`s3`\|`sharepoint`\|`pyfilesystem` | `fs` | indexer | Source type — see [Sources](#sources) below. |
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
| **pyfilesystem** `sources[].fs_url` | str | — (required) | indexer | Any PyFilesystem URL: `ftp://…`, `ssh://…`, `webdav://…`, `zip://…`, `osfs://…` (extra: `serviette[pyfilesystem]`; some protocols need a driver package). |
| **pyfilesystem** `sources[].path` | str | `""` | indexer | Path inside the opened filesystem (recursive). |
| **pyfilesystem** `sources[].refresh_interval` | float | `30.0` | indexer | Seconds between scans (streaming mode). |
| `parser` | list of rules | keyless-first defaults | indexer | Routing rules `{match: ["*.pdf"], type: docling, options: {...}}`, first match wins; unmatched files use built-in defaults (text→utf8, pdf→docling/pypdf, office→unstructured, images→paddle_ocr, audio→whisper if `OPENAI_API_KEY`, video→twelvelabs_video if `TWELVELABS_API_KEY`, else skip+warn). Types: `utf8`, `pypdf`, `docling`, `unstructured`, `paddle_ocr`, `vision_image`, `vision_slide`, `whisper`, `twelvelabs_video`, `skip`. Changing routing is fingerprint-guarded. |
| `vector_db.type` | `duckdb`\|`pgvector`\|`milvus`\|`qdrant`\|`chroma`\|`weaviate`\|`pinecone`\|`mongodb` | — (required) | both | Vector database backend (see [Backends](#vector-database-backends)). |
| **duckdb** `vector_db.path` | str | — (required) | both | Path to the DuckDB database file. |
| **duckdb** `vector_db.table` | str | `serviette_embeddings` | both | Target table. |
| **pgvector** `vector_db.connection_string` | str | — (required) | both | `postgresql://user:pass@host/db`. |
| **pgvector** `vector_db.table` | str | `serviette_embeddings` | both | Target table. |
| **milvus** `vector_db.uri` | str | — | both | Milvus URI, e.g. `http://localhost:19530`. |
| **milvus** `vector_db.host` / `.port` | str / int | `localhost` / `19530` | both | Used if `uri` is omitted. |
| **milvus** `vector_db.collection` | str | `serviette_embeddings` | both | Milvus collection. |
| **qdrant** `vector_db.host` | str | `localhost` | both | Qdrant host. |
| **qdrant** `vector_db.rest_port` / `.grpc_port` | int | `6333` / `6334` | server / indexer | REST (server) and gRPC (indexer) ports. |
| **qdrant** `vector_db.api_key` | str | — | both | Qdrant Cloud API key. |
| **qdrant** `vector_db.collection` | str | `serviette_embeddings` | both | Auto-created (cosine) if missing. |
| **chroma** `vector_db.host` / `.port` | str / int | `localhost` / `8000` | both | Chroma server address. |
| **chroma** `vector_db.tenant` / `.database` | str | Chroma defaults | both | Multi-tenant selectors. |
| **chroma** `vector_db.collection` | str | `serviette_embeddings` | both | Auto-created (cosine `hnsw:space`). |
| **weaviate** `vector_db.http_host` / `.http_port` | str / int | `localhost` / `8080` | both | HTTP endpoint. |
| **weaviate** `vector_db.grpc_host` / `.grpc_port` | str / int | http_host / `50051` | server | gRPC endpoint (v4 client). |
| **weaviate** `vector_db.api_key` | str | — | both | API key. |
| **weaviate** `vector_db.collection` | str | `ServietteEmbeddings` | both | Auto-created (capitalized name). |
| **pinecone** `vector_db.index_name` | str | — (required) | both | Auto-created (serverless, embedder's dimension, cosine). |
| **pinecone** `vector_db.api_key` | str | `${PINECONE_API_KEY}` | both | Pinecone API key. |
| **pinecone** `vector_db.namespace` | str | `""` | both | Optional namespace. |
| **mongodb** `vector_db.connection_string` | str | — (required) | both | `mongodb+srv://...` for Atlas. |
| **mongodb** `vector_db.database` / `.collection` | str | — / `serviette_embeddings` | both | Target collection (auto-created). |
| **mongodb** `vector_db.vector_index` | str | `vector_index` | server | Atlas `vectorSearch` index name. |
| `embedder.type` | str | `openai` | both | Embedder family (see below). |
| `embedder.model` | str | provider default | both | Model name. |
| `embedder.api_key` | str | — | both | API key (or `${ENV}`). |
| `splitter.type` | `token_count`\|`recursive` | `token_count` | indexer | Chunking strategy. |
| `splitter.chunk_size` | int | `512` | indexer | Max chunk size (tokens). |
| `splitter.chunk_overlap` | int | `50` | indexer | Overlap (used by `recursive`). |
| `indexer.workers` | int | `1` | indexer | Worker **processes** (sharded via `pathway spawn`). Raise for large backfills — each worker carries its own embedding stack (~1 GB with local embeddings); the benchmarks run with `8`. |
| `persistence.enabled` | bool | `true` | indexer | See [Persistence](#5-persistence). |
| `persistence.backend` | `filesystem` | `filesystem` | indexer | Persistence backend. |
| `persistence.path` | str | `./persistence` | indexer | Persistence directory (also hosts the parse cache under `runtime_calls/`). Silent default — the wizard does not ask. |
| `server.host` / `server.port` | str / int | `127.0.0.1` / `8989` | server | Bind address. Loopback by default; set `0.0.0.0` explicitly to listen on all interfaces (containers, remote access) — see [Security](#security--exposing-the-server). |
| `server.serve_frontend` | bool | `true` | server | Serve the chat UI on `/` from the same port (API stays under `/api/v1`). |
| `server.cors_origins` | list[str] | `[]` (disabled) | server | Opt-in CORS allowlist for third-party browser frontends calling the API directly from another origin. serviette's own UIs never need it. Prefer exact origins over `*`. |
| `llm.type` | str | `openai` | server | LLM for `/rag` (omit to disable `/rag`). |
| `llm.model` | str | `gpt-4o-mini` | server | Chat model. |
| `llm.api_key` | str | — | server | API key (or `${ENV}`). |
| `frontend.host` / `frontend.port` | str / int | `127.0.0.1` / `3000` | frontend (standalone) | Bind address for the split-deployment chat UI. |
| `frontend.api_url` | str | `http://localhost:8989` | frontend (standalone) | Base URL of the API the standalone frontend proxies to — point it at the backend host in split deployments (frontend fleet / backend fleet). The address never reaches the browser. |
| `frontend.title` | str | `serviette` | both | Title shown in the chat UI (embedded and standalone). |

**Supported embedders** (from `pathway.xpacks.llm.embedders`): `openai`,
`litellm` (≈100 providers), `sentence_transformer` (fully local, no
credentials — install `serviette[local]`), `gemini`, `bedrock`. Every family has a
matching async client on the server side, so one `embedder` section serves both
components; make sure the indexer and server use the **same** model so vectors
are comparable.

### Sources

serviette only supports sources that offer Pathway's **`only_metadata`** read mode —
the graph holds just paths/identifiers and metadata (never the file bytes), and
the bytes are fetched on demand during parsing. That keeps the memory footprint
tiny (1 GB of PDFs stays a few KB in the pipeline). Today that means:

- **`fs`** — a local directory, watched recursively per `glob`.
- **`gdrive`** — a Google Drive folder or file. Create a Google **service
  account**, download its JSON key, share the target folder with the service
  account's email, and point `service_user_credentials_file` at the key. See the
  [Pathway Google Drive connector guide](https://pathway.com/developers/user-guide/connect/connectors/gdrive-connector/).
  Install the extra: `pip install "serviette[gdrive]"`.
- **`s3`** — an S3 bucket or key prefix, including S3-compatible stores
  (MinIO, DigitalOcean, Wasabi) via `endpoint` + `with_path_style: true`.
  Credentials fall back to the standard AWS chain when omitted. No extra
  needed (boto3 ships with Pathway).
- **`sharepoint`** — a SharePoint directory or file, authenticated with an
  app-registration certificate (`tenant`, `client_id`, `cert_path`,
  `thumbprint`). Requires a Pathway **Scale** license and
  `pip install "serviette[sharepoint]"`.

- **`pyfilesystem`** — anything the
  [PyFilesystem2](https://docs.pyfilesystem.org) library opens: FTP, SFTP,
  WebDAV, even ZIP/TAR archives — selected by `fs_url` (e.g.
  `ftp://user:pass@host/dir`, `zip://./docs.zip`). Install
  `pip install "serviette[pyfilesystem]"`; some protocols need their own driver
  package (`fs.sshfs`, `fs.webdavfs`).

You can list several sources of mixed types in one config.

### Vector database backends

Every backend is written through Pathway's **native `pw.io.*` output
connector** in snapshot/upsert mode keyed by the chunk id, so file additions,
edits and deletions become real inserts/updates/deletes in the store. The
server retrieves through each database's own vector-search API — there are no
linear scans in Python anywhere.

**Targets are created for you.** At startup the indexer runs
`prepare_backend`: it creates the missing table / collection / index for
every backend (pgvector table + HNSW index, Milvus collection, Chroma and
Weaviate collections, the Pinecone serverless index, the MongoDB Atlas
`vectorSearch` index), sizing the vector dimension from
`vector_db.embedding_dimension` when set, or by introspecting the configured
embedder otherwise. The schemas below are shown for reference — e.g. when a
DBA manages the database — not as required manual steps.

#### DuckDB (embedded — the zero-setup default)

```yaml
vector_db:
  type: duckdb
  path: ./embeddings.duckdb
  table: serviette_embeddings
```

No external service and no setup: the file and table are created automatically
(`pw.io.duckdb`, snapshot mode). Embeddings are stored as native `DOUBLE[]`
columns and queried **inside DuckDB** with `list_cosine_similarity` — a
vectorized, columnar scan that answers in milliseconds for local corpora.

Concurrency: DuckDB allows one read-write process *or* several read-only
processes per file — never both. serviette resolves this with Pathway's
`detach_between_batches`: the streaming indexer releases the file lock
between minibatches, and the server's short-lived read-only connections (with
a retry through brief lock windows) query it concurrently — live indexing and
serving work on one file. On older Pathway builds without the flag, serviette
warns and falls back to the hold-the-lock behavior (use `mode: static`
there).

#### pgvector

Auto-created (extension, table, HNSW index). Reference schema, e.g. for
`text-embedding-3-small` (1536 dims):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE serviette_embeddings (
  chunk_id  text PRIMARY KEY,
  text      text,
  metadata  jsonb,
  embedding vector(1536)
);
```

(The indexer writes the chunk primary key to the `chunk_id` column.)

#### Milvus

Auto-created: a collection with a VARCHAR primary key, `text` (VARCHAR),
`metadata` (JSON) and an `embedding` (FLOAT_VECTOR) field of the embedder's
dimension, AUTOINDEX with the `COSINE` metric.

#### Qdrant

Nothing to create: the indexer (`pw.io.qdrant`, gRPC) auto-creates the
collection with the cosine metric on first write. The chunk's `text` and
`metadata` travel in the point payload; the server queries over REST.
Set `api_key` for Qdrant Cloud.

#### ChromaDB

Auto-created with the cosine distance (`hnsw:space`). The chunk text is
stored as the Chroma document and the source metadata as a JSON string
(Chroma metadata values must be scalars).

#### Weaviate

Auto-created (capitalized name, e.g. `ServietteEmbeddings`; vectorizer `none`).
The chunk vector is stored as the object vector; `text` / `metadata` (JSON
string) / `chunk_id` become properties. The server's v4 client needs both the
HTTP and gRPC ports.

#### Pinecone

Auto-created: a serverless index (`vector_db.cloud` / `.region`) with the
embedder's dimension and the cosine metric. The `text` and JSON-string
`metadata` are stored as record metadata. The API key comes from `api_key`
or `$PINECONE_API_KEY`.

#### MongoDB Atlas Vector Search

The indexer writes one document per chunk (`pw.io.mongodb`, snapshot mode).
The collection and the Atlas `vectorSearch` index (default name
`vector_index`, cosine, `numDimensions` from the embedder) are auto-created;
on Atlas Local the indexer rides out the search service's slow boot with
retries. The server retrieves with `$vectorSearch`.

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

## 6. Parsing & modalities

serviette never implements its own parsers — each file is routed by extension to
a `pathway.xpacks.llm.parsers` parser. Every format is enabled by default,
preferring parsers that need **no API key**, and modalities whose only parser
requires an absent key are skipped with a single clear warning (never a
crash):

- text / Markdown → as-is;
- `.pdf` → `DoclingParser` (layout-aware, tables — `serviette[docling]`), falling
  back to `PypdfParser`;
- Office & co. (DOCX, PPTX, XLSX, HTML, EML, CSV, RTF, EPUB, …) →
  `UnstructuredParser`;
- scanned images (PNG, JPG, TIFF, …) → `PaddleOCRParser` (`serviette[ocr]`);
- audio (MP3, WAV, …) → Whisper, enabled when `OPENAI_API_KEY` is set;
- video (MP4, WebM, MOV, …) → TwelveLabs Pegasus (a searchable text
  description), enabled when `TWELVELABS_API_KEY` is set.

The routing is overridable per file pattern via the `parser:` config section
(first matching rule wins; unmatched files use the defaults above) — e.g.
route images to a vision model instead of OCR, or set a custom video prompt.
Routing changes are fingerprint-guarded like splitter changes. Expensive
parses (video) are cached on disk, so restarts cost nothing.

Still not supported: encrypted / password-protected PDFs; Google-native files
(Docs/Sheets/Slides) from a `gdrive` source — they have no binary download
and would need an export step (TODO); any format the installed `unstructured`
version cannot handle. Unsupported files are skipped and produce no chunks.

---

## 7. Architecture

<p align="center">
  <img src="assets/architecture.svg" alt="serviette architecture: sources feed the Pathway indexer, which writes through Pathway's native connectors into one of 8 vector databases; the stateless server embeds queries, searches the database and serves the chat UI and the /api/v1 REST API" width="100%">
</p>

Inside the indexer, documents move through a Pathway dataflow — every stage
below is an incremental operator, so a change to one file re-runs only that
file's slice of the graph:

<p align="center">
  <img src="assets/indexer-pipeline.svg" alt="The indexer's Pathway dataflow: read in only_metadata mode (paths, never bytes) → parse UDF with parser routing and an LRU disk cache under the persistence directory → split into chunks → embed → snapshot sink emitting upserts and deletes keyed by chunk id into the vector database. A deleted file is retracted by replaying its memoized parse output negated." width="100%">
</p>

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
  host than the API, `serviette frontend` is a separate, stateless web tier that
  serves the same chat page and proxies to the API at `frontend.api_url`
  server-side (the API address never reaches the browser).
- **Indexer.** Runs as its own long-lived process and shards across worker
  processes with one config line (`indexer.workers`, via `pathway spawn`) —
  the published benchmarks run with 8; its persistence lets it resume
  without re-embedding.
- **Vector database.** Scaling and replicating the vector DB itself is the
  user's responsibility and follows that database's own guidance. (The embedded
  DuckDB backend is the exception: it lives in one file next to the indexer and
  is meant for local / single-node setups.)

Engine observability: with `indexer.monitoring_http_port` set, every worker
process serves the Pathway engine's built-in endpoints on
`127.0.0.1:(port + worker index)` — `GET /metrics` (Prometheus: input/output
latency gauges, per-operator row counters) and `GET /status`.

---

## Security & exposing the server

The API server binds to **127.0.0.1 by default**: on a laptop, `serviette up`
serves only you. It deliberately ships **no built-in authentication** —
that is a perimeter concern, and a reverse proxy does it better than any
hand-rolled API key (TLS, key rotation, SSO, rate limits, audit logs come
for free). This is the norm for tools of this class: Qdrant self-hosted
ships with auth off, Ollama has none at all — and internet scans keep
finding thousands of Ollama instances exposed by a careless `0.0.0.0`.
Don't join them.

To serve beyond localhost:

1. Set the bind address explicitly — it is an intentional decision:

   ```yaml
   server:
     host: 0.0.0.0     # containers / remote access; keep 127.0.0.1 otherwise
     port: 8989
   ```

2. Put an authenticating reverse proxy in front. A complete example with
   [Caddy](https://caddyserver.com) (TLS is automatic):

   ```
   rag.example.com {
       basic_auth {
           # generate the hash with: caddy hash-password
           alice $2a$14$...bcrypt-hash...
       }
       reverse_proxy 127.0.0.1:8989
   }
   ```

   With this in place serviette itself can stay on 127.0.0.1 — only the proxy
   is exposed. Any equivalent (nginx + auth_basic, oauth2-proxy, Cloudflare
   Access, a VPN like Tailscale) works the same way.

**CORS** is deliberately not emitted by default: serviette's own UIs never
need it (the embedded chat is same-origin; the standalone frontend proxies
server-side, so `frontend.api_url` may point at a backend on another host —
or another fleet — without any browser-visible cross-origin traffic), and
the absence of CORS headers is what stops arbitrary web pages from reading
responses off localhost instances. If you build your own browser frontend
that calls the API directly from another origin, opt in with an explicit
allowlist:

```yaml
server:
  cors_origins: ["https://app.example.com"]   # never "*" unless you truly mean it
```

The engine's monitoring endpoints (`indexer.monitoring_http_port`) are
loopback-only by construction and cannot be exposed directly.

---

## Appendix: regenerating the demo GIF

The README demo GIF is **emulated** (rendered with Pillow — no terminal
recorder or browser needed), so it regenerates deterministically anywhere:

```bash
pip install pillow
python docs/generate_demos.py     # -> docs/assets/demo.gif
```

The scene (the Team-tier question, the `sed` price edit, the second answer)
is parameterized by constants at the top of the script.
