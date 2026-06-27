# vetosh

**A universal, no-code RAG server for any vector database — powered by the
[Pathway](https://pathway.com) Live Data Framework.**

Set up real-time Retrieval-Augmented Generation over your own documents without
writing any code. Point vetosh at a folder, pick a vector database and an
embedder in a YAML file, and run a few commands.

<p align="center">
  <img src="docs/assets/demo.gif" alt="vetosh: CLI walkthrough then the web chat UI" width="100%">
</p>
<p align="center"><em>From zero to a live RAG stack — <code>quickstart → indexer → server → frontend</code> — then chat with your documents.</em></p>

```bash
pip install "vetosh[openai]"

vetosh quickstart                       # interactive config wizard
vetosh indexer  --config config.yaml    # watch files → embed → vector DB (live)
vetosh server   --config config.yaml    # FastAPI /retrieve and /rag endpoints
vetosh frontend --config config.yaml    # web chat UI → http://localhost:3000
```

```bash
curl -X POST http://localhost:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query": "how does persistence work?", "k": 5}'
```

## Highlights

- **No code.** Configure everything in one YAML file (or generate it with
  `vetosh quickstart`).
- **Any vector DB.** pgvector and Milvus out of the box.
- **Live & incremental.** Built on Pathway: additions, edits and deletions are
  reflected in the vector DB in real time. Sources are read in `only_metadata`
  mode, so a large corpus stays tiny in the pipeline.
- **Multiple sources.** Local filesystem and Google Drive (both `only_metadata`),
  mixed freely in one config.
- **Reuses `pathway.xpacks.llm`.** Parsers, splitters and embedders are used
  as-is — vetosh implements none of its own.
- **Decoupled & scalable.** Indexer, API server and web frontend are independent
  processes — put each on its own worker. The frontend proxies to the API at a
  configurable `api_url`; the server scales horizontally; they share only the
  vector DB.
- **Web chat UI.** `vetosh frontend` serves a clean ChatGPT/Claude-style chat
  page with grounded answers and expandable sources.
- **Free Pathway license.** One click at
  <https://pathway.com/framework/get-license>.

## Requirements

Python ≥ 3.10 (the minimum supported by Pathway).

## Documentation

Full installation, quickstart, configuration reference, persistence,
architecture and scaling notes live in **[docs/README.md](docs/README.md)**.

## Development

```bash
pip install -e ".[dev,openai]"
pytest -m "not slow"            # fast unit tests (no Pathway, no services)
pytest -m "slow and not integration"   # end-to-end indexer tests (spin up Pathway)
pytest -m integration          # real-database tests (see below)
pytest                         # everything
```

### Integration tests (real databases)

The `integration`-marked tests run the full indexer against a **real** vector
database and query it back through the production accessor:

- **pgvector** — `tests/test_integration_pgvector.py` spins up a throwaway
  `pgvector/pgvector` **Docker container** (via `tests/dockerutil.py`), creates
  the table, indexes documents, and verifies retrieval and deletion. It skips
  automatically if Docker is unavailable.
- **Milvus** — `tests/test_integration_milvus.py` uses the embedded **Milvus
  Lite** engine (`milvus-lite`), exercising the same `pw.io.milvus.write` sink
  and async accessor as a real server. It skips if `milvus-lite` is not
  installed.

`tests/dockerutil.py` is a small, dependency-free helper (Docker CLI via
subprocess) for spinning up throwaway DB containers — the same pattern extends to
a full Milvus standalone container if you prefer that over Milvus Lite.

## License

See [LICENSE](LICENSE).
