# FRAMES benchmark

End-to-end evaluation of serviette's retrieval quality on
[FRAMES](https://arxiv.org/abs/2409.12941) (Google, 2024): 824 multi-hop
questions whose answers must be assembled from 2–15 English Wikipedia
articles.

> **Start with [REPORT.md](REPORT.md)** — the full technical report:
> methodology, statistics, analysis, limitations. This README is the
> operational companion (fidelity summary, headline numbers, how to run);
> [results/RESULTS.md](results/RESULTS.md) holds the raw data tables.

The methodology tracks the paper wherever technically possible and
documents every deviation below.

## Fidelity to the paper

| Aspect | Paper | This benchmark |
|---|---|---|
| Questions | 824, `google/frames-benchmark` | Same dataset, revision pinned in `fetch_dataset.py` |
| Corpus | Full English Wikipedia, TFDS `wikipedia/20230601.en` | **Same dump, indexed in full**: 5.22M articles → 12.07M chunks — every headline number is measured against it. A small working corpus (the ~2.5k gold articles + a deterministic ~78k distractor sample) exists alongside for fast feature ablations only. Gold articles whose canonical dump title differs from the link (redirects) are fetched via the Wikipedia API **at the snapshot date**; a handful of gold links point at articles created *after* the snapshot — absent here exactly as they were absent in the paper's corpus |
| Retrieval unit | Whole articles, BM25, n_docs ∈ {2, 4} | 512-token chunks; k=8 chunks ≈ the same context budget as BM25\@4 short articles |
| Retrieval metric | Gold-article recall in context (0.12–0.15 for their BM25) | Same metric, `run_retrieval_eval.py` — directly comparable, and free of LLM cost |
| Headline metric | Accuracy via LLM autorater | Same |
| Judge | Gemini-Pro-1.5, prompt in paper Appendix B (0.96 acc. vs humans) | **Same prompt** (`judge_prompt.txt`), gpt-4o-mini at temperature 0 (validated at 94% agreement with gpt-4o) — Gemini-Pro-1.5 is retired |
| Generator | Gemini-Pro-1.5-0514 | Configurable (retired model can't be reproduced). Every run reports its generator; the controlled baseline below anchors comparisons |
| Multi-step baseline | 5 steps × 5 queries × 10 docs → 0.66 | serviette's `rag.adaptive` + `rag.decompose` are the single-shot analogue (documented as *analogous, not identical*) |

**The controlled baseline:** because the paper's generator is unavailable,
cross-paper numbers are context, not conclusions. The primary comparison is
internal and fully controlled — serviette with all retrieval features off vs.
feature configurations, on the identical corpus, generator, judge, and
prompts. Only the retrieval layer varies.

## Results

Headlines, measured on the paper's full Wikipedia dump under the paper's
permissive protocol (full analysis and statistics in
[REPORT.md](REPORT.md)):

- **Adding serviette to gpt-5 yields a paired +5.2 pp over the same model
  without retrieval** (z = 4.1 on 824 shared questions); under grounding,
  the adaptive-retrieval flag adds **+11.5 pp** (z = 7.5). For context, the
  absolute score (73.7%) exceeds every number in the paper, including its
  5-step agent (66.0%) and oracle (72.9%) — a comparison dominated by
  generator strength, which the report isolates.
- **Retrieval recall on the identical corpus: 0.50 vs 0.15** for the
  paper's BM25 (0.21 for our reproduction of it) — measured with the
  paper's own metric, no generator involved.
- A stricter **grounded** regime (answers only from documents — how
  serviette ships for private corpora) is reported separately in
  RESULTS.md and is deliberately not compared against the paper's numbers.

## Layout

```
fetch_dataset.py        questions.jsonl from HF (revision-pinned)
extract_corpus.py       corpora from the TFDS 2023-06-01 dump (run in a TF venv)
fetch_gold_fallback.py  the few redirect-titled gold articles, at snapshot date
configs/                indexing + evaluation configurations
run_retrieval_eval.py   phase 0: article recall, zero LLM cost
run_bench.py            end-to-end: /rag + Appendix-B judge
results/                one jsonl per run: summary line + per-question rows
```

## Runbook

```bash
export FRAMES_DATA=/path/to/data      # big; keep outside the repo
python fetch_dataset.py --out $FRAMES_DATA

# corpus (TF venv: pip install tensorflow-cpu tensorflow-datasets)
python extract_corpus.py --questions $FRAMES_DATA/questions.jsonl --out $FRAMES_DATA
python fetch_gold_fallback.py --missing $FRAMES_DATA/gold-missing.txt \
    --out $FRAMES_DATA/corpus-gold

# long-lived Qdrant (index survives restarts; re-runs never re-embed)
docker run -d --name frames-qdrant -p 6343:6333 -p 6344:6334 \
    -v $FRAMES_DATA/qdrant:/qdrant/storage qdrant/qdrant

serviette indexer --config configs/base-e5small.yaml       # working corpus (~1.5 h)
serviette indexer --config configs/base-e5small-full.yaml  # full dump (~75 h)
serviette server  --config <base + overlay>                # per configuration

python run_retrieval_eval.py --questions $FRAMES_DATA/questions.jsonl \
    --k 8 --label <config-label>                       # free
OPENAI_API_KEY=... python run_bench.py \
    --questions $FRAMES_DATA/questions.jsonl --k 8 --label <config-label>
```

Re-runs are cheap by construction: corpora and the Qdrant index persist, so
changing a server-side setting (hybrid, reranker, `rag.*`) and re-evaluating
takes minutes, not hours. Only changing the embedder or the chunking requires
re-indexing (into a fresh collection, so the old one stays usable).

## Evaluation protocol

1. **Phase 0 — retrieval ablations, free.** Rank all retrieval
   configurations by article recall\@k. No generation, no judge.
2. **Phase 1 — cheap generation sweep.** The shortlist runs end-to-end with
   an inexpensive generator to rank configurations by accuracy.
3. **Phase 2 — headline runs.** The best configuration and the
   features-off baseline run with the strongest generator; the judge stays
   fixed. Reported: accuracy, article recall, generator, cost.
```
