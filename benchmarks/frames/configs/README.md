# Bench configurations

`base-e5small.yaml` indexes both corpora into the long-lived Qdrant container.
Evaluation configs are the same file plus a **server-side overlay**: retrieval
features live in the `vector_db` / `reranker` / `rag` sections and need no
re-indexing — restart only `serviette server` with a different config to
switch configurations. Overlays used by the sweep (append to the base):

| Label | Overlay |
|---|---|
| `vector` | — (base as is) |
| `hybrid` | `vector_db.hybrid: true` |
| `hybrid-mmr` | + `rag: {mmr: {candidates: 24, diversity: 0.3}}` |
| `hybrid-ce-rerank` | + `reranker: {type: cross_encoder, candidates: 30}` |
| `hybrid-decompose` | + `rag: {decompose: {max_subqueries: 4}}` (needs `llm`) |
| `best-adaptive` | best of the above + `rag.adaptive` (e2e phases only) |

Generation phases add an `llm` section (e.g. `type: openai`, `model: gpt-4o`,
`temperature: 0`). The `e5base` variant swaps the embedder model to
`intfloat/e5-base-v2` and the collection to `frames_e5base` (separate
collection = both embedders stay indexed side by side).
