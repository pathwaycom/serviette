# Real-time data indexing benchmark

Measures the two claims that matter for the vetosh indexer:

1. **Indexing speed** — wall-clock time to index a document collection end to
   end (parse → chunk → embed → write to the vector DB).
2. **Flat memory** — the indexer's memory consumption does **not** grow with
   the size of the input collection (`only_metadata` sources keep paths, not
   bytes, in the pipeline). The output is a memory-over-time plot and the
   physical peak memory (PSS: shared pages counted once, not per process).

Everything runs locally and costs nothing: embeddings are computed by a local
sentence-transformers model (`all-MiniLM-L6-v2`, 384 dims) — **no external
API calls**.

## Setup

Three containers via docker-compose:

- **qdrant** — the vector store. Chosen deliberately: HNSW builds
  incrementally on upsert (fast indexing), it is designed for concurrent
  search-while-writing, and the Pathway sink keys points internally (no user
  primary key), so writes parallelize across Pathway workers.
- **indexer** — `vetosh indexer` over the dataset (static mode: the run ends
  when the collection is fully indexed, so wall-clock = indexing time).
- **server** — `vetosh server`; used after indexing to replay the accuracy
  questions through `/api/v1/retrieve`.

Memory is sampled from `/proc/1/status` inside the indexer container every
few seconds: `VmRSS` and `Pss` for the time series (PSS divides pages shared
between workers — e.g. the PyTorch libraries — proportionally, so its sum is
the honest physical footprint), `VmHWM` (kernel-tracked high-water
mark) for the exact peak — no sampling artifacts.

## Dataset

`prepare_dataset.py` downloads Wikipedia parquet shards from Hugging Face
(public, fast CDN) and writes one `.txt` file per article until the target
size is reached. It also samples distinctive articles into `questions.json`;
each question must retrieve its own article (top-k accuracy sanity check).

## Running

```bash
python prepare_dataset.py --size-mb 100     # also: 1000, 3000, 10000
python run_bench.py --size 100mb
```

Outputs land in `results/`: `<size>-memory.png` (memory over time),
`<size>-memory.csv`, `<size>-summary.json` (duration, MB/min, chunk count,
peak RSS/PSS, retrieval accuracy).

To make the flat-memory point, run the same benchmark at increasing sizes
(100mb → 1gb → 3gb → 10gb) and compare `peak_rss_mb` across summaries: the
corpus grows 100×, the peak stays in the same band.
