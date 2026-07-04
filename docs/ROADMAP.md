# vetosh — status & roadmap

*Last updated: 2026-07-04.*

## Where we are

vetosh is a no-code RAG stack built as a **showcase of Pathway connectors**:
one YAML file turns Pathway's native `pw.io.*` output connectors + `xpacks.llm`
parsers/splitters/embedders into a live document-indexing pipeline, a retrieval
API and a chat UI — with zero application code written by the user.

**Vector-DB backends (8):** DuckDB (embedded, zero-setup default), pgvector,
Milvus, Qdrant, ChromaDB, Weaviate, Pinecone, MongoDB Atlas Vector Search.
All written through native Pathway connectors in snapshot mode — source
adds/edits/deletes become real upserts/deletes in the store.

**Embedders (5 families, both sides):** OpenAI, LiteLLM (≈100 providers),
SentenceTransformers (fully local), Gemini, Bedrock.

**Modalities today:** text (`.txt`/`.md`), PDF (pypdf), and everything
`unstructured` handles (DOCX, PPTX, XLSX, HTML, EML, CSV, RTF, EPUB, …).
Available in the xpack but not yet wired in: images/slides via vision LLMs
(`DoclingParser`, `ImageParser`, `SlideParser`), OCR (`PaddleOCRParser`),
audio via Whisper (`AudioParser`). No video.

**The differentiator**: *continuous incremental sync with correct deletion
semantics into an external vector DB of your choice*. Competing tools re-index
on a schedule or on upload; vetosh watches sources and propagates a file edit
or removal to the vector DB within seconds, without re-embedding unchanged
content (Pathway persistence + embedder cache). That story is exactly one YAML
file — because the machinery is Pathway itself.

## Competitive landscape (as of 2026-07)

| Tool | Stars | External vector DBs | Continuous sync incl. deletes |
|---|---|---|---|
| **vetosh** | — | **8, native connectors** | **yes — streaming watch, real deletes** |
| Dify | 147k | ~20, but captive KB storage | no (manual / triggered re-sync) |
| RAGFlow | 84k | none (bundled ES/Infinity) | no (batch upload) |
| AnythingLLM | 62k | ~10 | partial beta: per-file hourly re-embed, desktop-only |
| PrivateGPT | 57k | ~5 | no (bulk ingest; dormant) |
| LlamaIndex OSS | 51k | 40+ | deletes only on manual pipeline re-run |
| Onyx (ex-Danswer) | 31k | none (captive Vespa→OpenSearch) | closest: 30-min polling + periodic pruning |
| Haystack | 26k | ~27 | no (DIY pipelines, batch) |
| Airbyte vector dest. | 22k | ~7 | scheduled; deletes only from CDC DB sources |
| Unstructured | 15k | ~20 | scheduled incremental; no delete propagation |
| R2R | 8k | 1 (pgvector) | no (API-driven) |
| LlamaCloud (SaaS) | — | 7 sinks | near-match, but proprietary/managed |

The open slot vetosh occupies: **OSS, watches folders/Drive continuously,
propagates adds+edits+deletes in seconds, and writes into *your* vector DB**
— either captive-index tools (Onyx, RAGFlow, Dify) or batch external-DB
writers (Airbyte, Unstructured, LlamaIndex) cover only one half each. The
pitch to lead with: *"the only open-source tool where deleting a file
actually deletes its vectors — in any of 8 databases — live."*

**Known constraints:**

- The new connectors (duckdb/qdrant/chroma/weaviate/pinecone) need a Pathway
  release that includes them; today vetosh runs against the develop tree.
- ~~DuckDB single-writer~~ — resolved: `pw.io.duckdb.write` ships
  `detach_between_batches`; a streaming indexer and the serving process now
  share the file (verified e2e).
- ~~targets must be pre-created~~ — resolved: `prepare_backend` creates
  tables/collections/indexes for all 8 backends at indexer startup.

---

## Category 1 — before showing to business colleagues

*Goal: a flawless 5-minute "look what falls out of Pathway connectors for
free" demo. Polish the first-run experience; make the live-sync magic visible.*

1. ~~**`vetosh up`**~~ — DONE: dumb supervisor, both processes children,
   uniform streaming for all backends; DuckDB live-serves through
   `detach_between_batches`.
2. ~~**Bundled demo corpus + script** (`vetosh demo`)~~ — DONE: Lumina Coffee
   corpus, zero-to-chat one command, pricing-edit scene verified e2e.
3. ~~**Fully local no-key path as the wizard default**~~ — DONE: DuckDB +
   local embeddings first, minimal question path, silent defaults
   (port 8989, persistence, splitter).
4. ~~**Re-record the README GIF**~~ — DONE (2026-07-04): quickstart →
   `vetosh up` → live edit → chat; regenerate with
   `python demos/generate_demos.py`.
5. ~~**Index statistics endpoint + frontend liveness**~~ — DONE:
   `/api/v1/stats` + "indexed N s ago" in the chat header.
6. **Friendly failure modes + `vetosh doctor`** — REMAINS: validate config,
   ping DB, check embedder dimension vs index, check parser deps. (Partial:
   fingerprint diffs, license hints, S3 region validator already done.)
7. **Smoke-test the demo path in CI** — test written
   (`tests/test_demo_path.py`), needs a green run + CI wiring.

*(moved to Category 2: docker-compose live-sync demo — no longer a demo
blocker since DuckDB live-serves; it's now an example deployment.)*

## The showcase rule

vetosh exists to demonstrate the Pathway framework. Every roadmap item must
state what Pathway/xpack functionality it reuses directly; features that
would be standalone code written *beside* the framework are flagged ⚠️/❌ and
held back unless separately justified.

## Category 2 — before showing to the external world

*Goal: credibility. Anyone can install it, run it against their own stack, and
nothing embarrasses us.*

1. **Depend on a released Pathway** that ships the new connectors (today:
   develop build). Blocks everything — PyPI users cannot run the new backends.
2. **Integration-test matrix for all 8 backends** (docker-compose services in
   CI; Pinecone Local for pinecone; Atlas local dev image for mongodb).
3. **PyPI release automation** + versioning + changelog; `pip install vetosh`
   must just work.
4. **Server hardening**: optional API-key auth, CORS config, rate limiting,
   request size limits, timeouts.
5. ~~**Auto-create targets**~~ — DONE: `prepare_backend` creates
   pgvector/Milvus/Chroma/Weaviate/Pinecone/MongoDB targets with the right
   dimension (config → embedder introspection fallback).
6. **Observability**: structured logs, `/metrics` (Prometheus), indexing lag
   gauge.
7. **Streaming `/rag` responses** (SSE) in server + frontend.
8. **More modalities via config**: `parser:` section exposing Docling
   (PDF+tables+images), ImageParser, AudioParser, PaddleOCR from the xpack.
9. ~~**S3 / SharePoint sources**~~ — DONE: fs/gdrive/s3/sharepoint wired
   into `sources:` with `only_metadata` + `max_backlog_size`; integration
   tests against MinIO.
9b. **docker-compose example deployment** (qdrant + vetosh services) — moved
   from Category 1.
10. **Optional reranker** — agreed 2026-07-04: `reranker:` config section,
    server fetches k×3 candidates, reranks to top-k in `/retrieve`/`/rag`;
    showcase = benchmark accuracy before/after at identical indexing cost.
    ⚠️ Showcase check pending: xpack `rerankers` are graph UDFs; verify they
    are callable standalone in the thin server. If not, this is bespoke
    code — decide explicitly before building.
11. **`pyfilesystem` source** — reuses `pw.io.pyfilesystem` (only_metadata):
    one new `sources:` type = live watch over SFTP/FTP/WebDAV/ZIP. ✅ pure
    Pathway showcase.
12. **`/metrics`** — reuses the engine's monitoring HTTP server
    (`pw.run(with_http_server=True)`), Prometheus format for free. ✅ pure
    Pathway showcase.
13. **Change alerts to Slack** — reuses `pw.io.slack` as a second graph
    output ("N documents re-indexed"); small, sells the streaming nature.
    ✅ pure Pathway showcase.
14. **Docs site**: per-backend tutorials, troubleshooting, architecture page;
    CONTRIBUTING + issue templates; choose and state the OSS license clearly.
15. **CI quality gates**: ruff, mypy, py3.10–3.13 matrix.

## Category 3 — after the public prototype

1. **MCP server mode** — ❌ held back by the showcase rule (2026-07-04
   audit): xpack `McpServer` serves from *inside* a Pathway graph, which
   contradicts vetosh's decoupled thin server; a FastMCP facade on our
   server would be standalone code reusing nothing from Pathway. Revisit if
   the xpack grows a graph-free serving mode, or accept as flagged bespoke.
2. **Adaptive RAG** as an opt-in `/rag` strategy — reuses the xpack
   strategy functions (`answer_with_geometric_rag_strategy`) called with our
   retrieved docs; ✅ xpack reuse (functions, not the graph server).
2b. **Stream sources** (Kafka/NATS/MQTT via `pw.io.*`) — documents as
   events; ✅ pure Pathway showcase. **Airbyte source** (`pw.io.airbyte`,
   hundreds of SaaS sources) — ✅ reuse but operationally heavy.
2c. **Elasticsearch output** (`pw.io.elasticsearch` + our accessor, same
   pattern as the existing 8 backends) — ✅ reuse; bridge to hybrid search.
3. **Evaluation harness** (RAGAS or similar) + per-backend latency/recall
   benchmarks; publish numbers.
4. **LEANN backend** once its reader path matures (today: write-only,
   full rebuild per commit).
5. **DuckDB ANN**: upstream a `FLOAT[n]` fixed-size array option in
   `pw.io.duckdb.write` so the VSS/HNSW index applies; today retrieval is a
   vectorized in-DB scan (fine at demo scale).
6. **Multi-collection / multi-tenant** serving; per-user corpora.
7. **Frontend growth**: chat history, source previews, auth.
8. **Kubernetes/Helm deployment** recipes; horizontal-scale guide.
9. **Video modality** (frame sampling + vision parsing) — no xpack support
   yet; needs design.
