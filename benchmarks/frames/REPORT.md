# Single-Shot Retrieval-Augmented Generation on FRAMES: A Controlled Evaluation of serviette

**Status: complete.** Two secondary measurements (the reasoning-effort
ablation and grounded weak-model rows) are deferred to future work — see
§7 and §8.

## Abstract

We evaluate serviette — a no-code, single-shot RAG server built on the
Pathway Live Data Framework — on
[FRAMES](https://arxiv.org/abs/2409.12941) (Krishna et al., 2024), an
824-question multi-hop benchmark over English Wikipedia. We reproduce the
paper's corpus exactly (the TFDS `wikipedia/20230601.en` dump, 5.22M articles,
12.07M chunks), adopt its autorater prompt, and measure under two explicitly
separated regimes: *permissive* (the paper's protocol — the generator may
combine retrieved context with parametric knowledge) and *grounded* (strictly
context-only — serviette's product default, deliberately stricter than the
paper). The system-attributable results: on the identical corpus and
metric, serviette's retrieval reaches **0.50 gold-article recall versus
0.15 published / 0.21 reproduced for the paper's BM25 baseline**; adding
serviette to gpt-5 under the paper's protocol yields a **paired +5.2 pp
over the same model without retrieval** (McNemar z = 4.1 on 824 shared
questions); and under grounding, refusal-triggered **adaptive retrieval —
one product configuration flag — lifts accuracy 41.3% → 52.8%** (z = 7.5),
the largest single effect we measure, anticipated by an independently
measured recall-vs-k curve. For context only: the absolute permissive
score, 73.7%, exceeds every number the paper reports, including its
five-step agentic pipeline (66.0%) and oracle (72.9%) — but most of that
gap belongs to the 2026 generator, whose no-retrieval baseline alone scores
68.5% on this 2024 benchmark; we therefore foreground paired deltas, not
absolute comparisons. Across our three-model ladder the paired gain is
significant at both ends (+3.8 pp weakest, +5.2 pp strongest) with no
measurable effect mid-tier; whether this is a genuinely non-monotonic
pattern is left as a hypothesis pending replication. A failure taxonomy
attributes 62% of remaining permissive errors to retrieval misses and
bounds the autorater's false-negative rate at ≈3 pp (false positives
unaudited). All corpora, configurations, raw per-question outputs, and
scripts are pinned and included.

## 1. Scope and claims

This report makes three claims and explicitly does not make several others.

**Claimed:**
1. On the paper's own corpus and retrieval metric, serviette's single-shot
   retrieval stack outperforms the paper's naive BM25 baseline ~2.4–3.3×
   (0.50 vs 0.15 published / 0.21 reproduced; metric-level uncertainty
   discussed in §5.1).
2. serviette's retrieval produces a significant paired gain on a frontier
   generator under the paper's protocol (+5.2 pp, z = 4.1), and adaptive
   retrieval under grounding produces the largest effect we measure
   (+11.5 pp, z = 7.5). The absolute permissive score exceeding every
   number in the paper is offered as context, not as a claim about the
   system — that comparison is dominated by generator strength (§5.2).
3. For a strictly grounded system the binding constraint is retrieval
   recall: 60% of grounded errors are refusals, the oracle ceiling sits
   ≈27 pp above the fixed-k configuration, and adaptive retrieval recovers
   42% of that headroom.

**Observed, not claimed:** paired retrieval gains are significant at both
ends of the model ladder and absent in the middle. The "non-monotonic in
generator strength" reading and its proposed mechanism are hypotheses: the
middle point is a single-run null result, and per-row sub-query generation
confounds reader strength with retrieval quality (§6.1).

**Not claimed:** superiority over the paper's *system* on matched generators
(their generator is retired; we bracket its strength instead — §5.2); any
number under the grounded regime being comparable to the paper (§4.3); that
FRAMES measures pure retrieval for frontier models (it does not — §6.1);
that the mid-tier null result establishes harm from retrieval (it is
statistically indistinguishable from no effect).

## 2. System under test

serviette (this repository) — a two-process RAG stack: a Pathway-based
indexer (parsing → 512-token chunking, 50 overlap → embedding → vector-DB
write-through) and a stateless retrieval/answer server. Configuration under
test:

- **Embedder:** `intfloat/e5-small-v2` (33M parameters, 384-d), local CPU,
  asymmetric `query:`/`passage:` prefixes. No embedding API calls anywhere.
- **Vector store:** Qdrant, cosine, HNSW; single named dense slot.
- **Retrieval features:** multi-hop query decomposition (`rag.decompose`,
  ≤4 sub-queries, round-robin fusion); adaptive context growth
  (`rag.adaptive`, k=8→16→32 on refusal; §5.4). In-process BM25 hybrid and
  MMR exist in the product but are **disabled here**: hybrid by its
  documented 5M-chunk cap (native sparse-vector hybrid is future work,
  §8), MMR to keep the tested configuration minimal.
- **Generation:** `/rag` endpoint, k=8 chunks (~4.1k tokens of context —
  matching the paper's BM25@4-articles budget).

## 3. Benchmark construction and fidelity

| Aspect | Paper | This evaluation |
|---|---|---|
| Questions | 824, `google/frames-benchmark` | identical; HF revision `58d9fb63` pinned |
| Corpus | full English Wikipedia, TFDS `wikipedia/20230601.en` | same dump: 5,222,443 articles ≥500 chars → 12,068,600 chunks |
| Gold articles | annotator-linked `wiki_links` | 2,453/2,477 present (99.03%); the 24 absent were created after the snapshot — absent for the paper as well. Redirect-titled links resolved via snapshot-date revisions; recall matching uses an alias map |
| Retrieval metric | gold-article recall in context | identical (article counted if ≥1 of its chunks retrieved — an optimistic reading vs. the paper's whole-article inclusion; noted in §7) |
| Judge | LLM autorater, Appendix-B prompt, 0.96 agreement w/ humans | same prompt verbatim; gpt-4o-mini, temperature 0. Validation: 94% agreement with a gpt-4o judge on a 100-question sample drawn from this benchmark's July predecessor run (same prompt and judge configuration, different answer set); noise additionally bounded on *this* run's answers via the failure audit, §6.3 |
| Generator | Gemini-Pro-1.5-0514 (retired) | three tiers (gpt-4o-mini / gpt-4o / gpt-5) bracketing the paper's generator by the naive anchor (§5.2) |

Corpus preparation is fully scripted (`fetch_dataset.py`,
`extract_corpus.py`, `fetch_gold_fallback.py`); distractor-free — the full
dump *is* the distractor set.

**The working corpus.** Alongside the full dump we maintain a small corpus
cut from the same snapshot: the 2,453 gold articles plus 77,961 distractor
articles selected by a deterministic title hash — 80,488 articles, 208,010
chunks, indexable in ~1.5 h versus 75 h for the full dump. It exists purely
for fast feature-ablation iteration (which retrieval features help, and in
what order); retrieval there is ~80× easier by gold base-rate, so **no
headline or cross-paper number is measured on it**. Where its results
appear (feature ranking in §5.3, the corpus-size check in §5.1, the hybrid
estimate in §8) they are labeled "working corpus" explicitly.

## 4. Methodology

### 4.1 Metrics
- **Article recall@k** — the paper's retrieval metric; generator-free.
- **Accuracy** — Appendix-B autorater verdicts over free-form answers.
- **Paired significance** — McNemar's test on shared questions for every
  retrieval-vs-naive comparison (n = 824 pairs); we report b (fixed by
  retrieval), c (broken by retrieval), z.

### 4.2 Cost/compute accounting
Embedding and retrieval are local and free. API usage is metered from
response objects for judge calls and for the direct-call rungs
(naive/oracle, `run_ladder.py`); generation issued *inside the serviette
server* was not usage-logged (a known instrumentation gap) and is estimated
from token arithmetic. Total API spend for every number in this report:
≈$128 (per-run figures in `results/*.jsonl` summaries). Indexing: 75 h on
96 CPU cores (8 Pathway workers × 12 threads, ~2,700 chunks/min). One
operational observation for practitioners: at equal API conditions,
per-question generation latency roughly doubled from the grounded to the
permissive regime (≈46 s → ≈96 s), indicating substantially longer
reasoning traces, billed as output tokens.

### 4.3 Two regimes, never mixed
**Grounded** (product default): the system prompt forbids outside knowledge;
insufficient context must produce a refusal. This isolates retrieval quality
from parametric memory — a reader can attribute grounded accuracy to the
retrieval stack. It is *stricter than the paper*: the paper's prompt imposed
no such restriction, which its own numbers prove arithmetically (47.4%
accuracy at 0.15 recall is unreachable under grounding).
**Permissive** (the paper's protocol): context plus the model's own
knowledge; the only regime whose accuracies may be set beside the paper's.

## 5. Results

### 5.1 Retrieval (generator-free, full dump)

Configuration labels: **vector** = plain dense retrieval (embed the
question, take the k nearest chunks by cosine similarity);
**vector+decompose** = the same, plus the LLM-generated sub-queries of §2
retrieving in parallel with the original question, results fused
round-robin.

| Method | Article recall@context | Full-coverage rate |
|---|---|---|
| BM25@4 whole articles (paper, published) | 0.12–0.15 | — |
| BM25@4 whole articles (our reimplementation) | 0.210 | 0.034 |
| serviette, vector only, k=8 | 0.284 | 0.078 |
| serviette, vector+decompose, k=8 (mini sub-queries) | 0.406 | 0.134 |
| serviette, vector+decompose, k=8 (gpt-5 sub-queries) | 0.497 | — |

**Recall as a function of context size** (mini sub-queries for the
decompose rows):

| k | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| vector | 0.232 | 0.284 | 0.334 | 0.386 | 0.428 |
| vector+decompose | 0.328 | 0.407 | 0.468 | 0.529 | 0.577 |

(The k=8 decompose cell, 0.407, is an independent re-run of the 0.406 row
in the table above; the 0.001 discrepancy is the observed re-run variance —
ANN search plus mini-generated sub-queries — and doubles as a measure of
retrieval-pipeline reproducibility.)

Recall grows ≈ logarithmically in k with no saturation by k=64, and
decomposition's advantage (+10–15 pp) is stable across the whole range —
the two mechanisms are complementary, not redundant. The curve also
anticipates the adaptive-retrieval result (§5.4): a refusal-triggered
escalation to k=32 operates at ≈0.53 recall instead of 0.41.

The BM25 rows involve no index and no part of serviette: they are computed
by a standalone two-pass streaming scorer (`run_bm25_baseline.py`) directly
over the corpus text files — Okapi BM25 (k1=1.5, b=0.75), whole articles,
the raw question as the query, exactly the paper's baseline setup. (BM25
needs only corpus term statistics; an index buys query speed, which a
one-shot batch evaluation does not need.)

Two observations. First, this BM25 reproduction validates the evaluation
pipeline: same method, same corpus, same metric land in the paper's
ballpark (0.21 vs 0.15; residual gap attributable to tokenizer/stemming
differences). Note the discrepancy's direction: our reimplementation
credits the baseline with a *stronger* result than its authors published,
and the headline multiplier is computed against that stronger 0.21 —
implementation uncertainty is priced against serviette, not for it.
Two metric-level biases, however, pull in opposite directions and do not
cancel: chunk-level coverage (≥1 chunk counts an article) reads *higher*
than the paper's whole-article inclusion, favoring serviette, while the
stronger reproduction deflates the multiplier, favoring the baseline. The
2.4–3.3× range therefore carries more uncertainty than its endpoints
suggest; the *ordering* — serviette well above BM25 on the same corpus —
is robust, the exact multiplier is indicative.
Second,
decomposition — one cheap LLM call — is worth +12 to +21 recall points on
multi-hop questions, and the quality of the decomposing model matters
(mini 0.406 → gpt-5 0.497).

*Corpus-size honesty check:* on an easy 80k-article working corpus the same
stack reads 0.62–0.74 recall while the same BM25@4 reads 0.508 — i.e., most
of any "5×" headline against the paper's 0.15 would be haystack size, not
stack quality. All cross-paper claims in this report therefore use the
full-dump numbers above.

### 5.2 End-to-end, permissive regime (the paper's protocol)

**Naive** (the paper's term) = no retrieval at all: the model receives the
bare question and answers from parametric memory alone — the floor against
which retrieval's contribution is measured. serviette = vector+decompose at
a fixed k=8 (single-shot; adaptive retrieval applies only to the grounded
regime, §5.4 — its escalation trigger is the grounded refusal marker, which
permissive answering rarely emits), full dump. Δ is the paired gain over
the same generator's naive row. Note an intentional design property:
each row is the *self-contained system* a user would deploy — decomposition
sub-queries are produced by the row's own generator, so retrieval quality
itself varies with the model (context recall column). Δ therefore measures
"what does adding serviette to this model do", not "same context, different
reader".

| Generator | Naive | serviette RAG | Context recall | Δ | McNemar |
|---|---|---|---|---|---|
| gpt-4o-mini | 32.9% | 36.7% | 0.404 | +3.8 pp | b=104, c=73, z=+2.3 (p<0.02) |
| gpt-4o | 49.9% | 47.8% | 0.421 | −2.1 pp | b=86, c=103, z=−1.2 (n.s.) |
| gpt-5 | 68.5% | **73.7%** | 0.497 | +5.2 pp | b=78, c=35, z=+4.1 (p<10⁻⁴) |
| *Gemini-Pro-1.5 (paper)* | *40.8%* | *BM25@4 47.4 · 5-step agent 66.0 · oracle 72.9* | *0.15 (BM25)* | | |

The paper's generator strength (naive 40.8%) is bracketed by our
gpt-4o-mini (32.9%) and gpt-4o (49.9%) rows.

### 5.3 Grounded regime (strictly context-only; not paper-comparable)

gpt-5, full dump, k=8: **41.3%** accuracy at 0.482 context recall. 60% of
errors are explicit refusals — recall, not reading, is the binding
constraint. The regime's ceiling was estimated two independent ways:

- **Oracle (all gold articles supplied, grounded prompt): 68.2%.**
- The free proxy — grounded accuracy on the 158 questions whose gold
  articles were fully retrieved: **68.4%**.

The two estimates are consistent, but the 0.2 pp agreement should not be
over-read: the proxy subset is biased toward easier questions (fully
retrievable gold sets), pushing it up; oracle contexts are truncated for
the longest questions, pushing the oracle down; and at n=158 the proxy's
CI is ±3.7 pp — the near-exact match is plausibly two opposing biases
meeting, not a precision result. Reading: perfect retrieval would lift
grounded accuracy from 41.3% to
≈68% (+27 pp of retrieval headroom); the residual ≈32% error at recall≈1
(24 post-snapshot gold articles remain unavailable even to the oracle, §3)
is the grounded reasoning/reading ceiling of the generator itself (note it
sits close to the model's permissive naive 68.5% — with complete evidence,
grounding costs nothing). Working-corpus grounded ablations (gpt-4o:
features-off 34.6% → full stack 43.8%) established feature ranking and are
reported in RESULTS.md.

### 5.4 Adaptive retrieval under grounding

`rag.adaptive` converts refusals into re-retrieval at k=16, then k=32 —
targeting exactly the dominant failure mode of §5.3. Its outcome was
anticipated by the independently measured recall-vs-k curve (§5.1:
escalation to k=32 should operate near 0.53–0.58 recall); we note the
sweep completed before this run did, but make no formal preregistration
claim.

**Result: 52.8% accuracy at 0.572 context recall** — versus 41.3% / 0.482
for fixed k=8. Paired on identical questions: adaptivity fixed 128 answers
and broke 33 (McNemar z = +7.5), the largest single effect measured in
this evaluation. Adaptive retrieval recovers 42% of the grounded regime's
retrieval headroom (41.3 → 52.8 of a 68.2 ceiling) at the cost of extra
context tokens only on the questions that refused — the single-shot
analogue of the paper's five-step agentic gain (+25.2 pp on its weak
generator), realized here inside one product configuration flag.

## 6. Analysis

### 6.1 Retrieval value across generator strength: two significant ends,
an open middle

What the data establish: significant paired gains at both ends of the
ladder (+3.8 pp weakest, +5.2 pp strongest) and no measurable effect for
the mid-tier model — a single-run null (z = −1.2), not an established
"wash-out". A non-monotonic ("U-shaped") pattern is one reading, and a
mechanistic story fits it — the weakest model substitutes reading for
memory it lacks, the frontier reasoner arbitrates between context and
memory (b=78 fixed vs c=35 broken), the mid-tier defers to incomplete
context — but we present it as a hypothesis, for two reasons. First,
establishing the shape requires the middle point to be a *replicated*
null, and generation-run variance is unmeasured (§7.1). Second, a
confound: each row's sub-queries come from its own generator, so gpt-5
enjoys both a stronger reader and stronger retrieval (recall 0.497 vs
0.404); the arbitration story is not separable, on these data, from
"better decomposition → better context". A cheap disentangling experiment
— one fixed set of sub-queries served to all three readers — is listed in
§8. What is not hypothetical: the paper's own +25.2 pp from five-step
agentic retrieval on a weak generator shows iteration is how weaker models
exploit retrieval, and the honest product case for RAG is corpora the
model has *not* memorized (fresh and private data), where the grounded
regime is the operative one and recall is decisive.

### 6.2 Failure taxonomy (permissive gpt-5, 217 errors, gpt-4o classifier)

| Category | Share | Reading |
|---|---|---|
| retrieval_miss | 61.8% | dominant; matches recall 0.497 — headroom lies in retrieval, not generation |
| reasoning_error | 19.8% | generator ceiling |
| judge_error | 11.5% | autorater graded an equivalent answer wrong |
| dataset_issue | 6.0% | consistent with the authors' own 5.5% staleness filtering |

### 6.3 Autorater error: a one-sided estimate
The failure audit (§6.2) examined only questions the judge marked wrong, so
by construction it can detect only *false negatives* — correct answers
misgraded: 25/824 = 3.0%. False positives (wrong answers credited) were
not audited at all, so no two-sided noise bound exists, and headline
accuracies cannot be called conservative on this evidence alone. The
auditor is moreover itself an LLM, whose errors may correlate with the
judge's. The 94% mini↔gpt-4o agreement (measured on the predecessor
answer set, §3) is consistent with a low-single-digit error rate; a proper
bound requires auditing a sample of *passed* verdicts — listed in §8.

## 7. Limitations

1. **Generator non-determinism and run variance:** gpt-5 rejects
   temperature control; each accuracy is a single run (n=1 per
   configuration; per-question CI ±3.4 pp). Re-run variance was measured
   only for the retrieval pipeline (±0.001 recall, §5.1) — never for
   generation, which weakens every conclusion resting on small deltas,
   most of all the mid-tier null (§6.1).
2. **Judge is a small model** (validated, bounded, but not human).
3. **Dataset contamination:** FRAMES (2024, Wikipedia) is inside 2026
   models' training data; permissive numbers partly measure memory. The
   regime split addresses attribution, not contamination itself.
4. **Chunk-level recall is optimistic** versus the paper's whole-article
   inclusion (a retrieved chunk may miss the needed fact).
5. **Hybrid and MMR disabled** in the tested configuration (§2). Grounded
   weak-model rows on the full dump are left to future work; the grounded
   regime is therefore characterized on gpt-5 only, plus the working-corpus
   gpt-4o ablations.
6. Oracle context truncated at 400k characters for the longest questions.

## 8. Future work

In rough priority order:

1. **Native sparse-vector (BM25) hybrid at full-dump scale.** The upstream
   connector support has landed in Pathway
   ([schema-driven named dense+sparse vectors in `pw.io.qdrant.write`](https://github.com/pathwaycom/pathway/commit/972e56cd1b),
   [sparse records in `pw.io.pinecone.write`](https://github.com/pathwaycom/pathway/commit/fdbedb1c58));
   the serviette integration and a 75 h re-index are scheduled separately.
   On the 80k working corpus the BM25 leg was worth +4 recall points; the
   full-dump gain is an open measurement.
2. **Replication of the model ladder** — repeated runs per configuration
   (generation variance) and grounded rows for the weaker models: the
   grounded regime, the most product-relevant one, is currently
   characterized on one model in one run.
3. **Disentangling the ladder confound** — one fixed set of sub-queries
   (e.g., gpt-5's) served to all three readers, separating reader strength
   from retrieval quality (§6.1).
4. **A two-sided autorater bound** — auditing a sample of *passed*
   verdicts, ideally with a non-LLM (human) spot check (§6.3).
5. **The reasoning-effort ablation** — the `llm.reasoning_effort` product
   knob against the latency observation of §4.2.

## 9. Reproducibility

Everything needed to re-run: `benchmarks/frames/` scripts (data fetch with
pinned revisions, corpus extraction from the public TFDS bucket, sweep
orchestrators, this report's numbers as raw per-question JSONL in
`results/`), configs (including ports and worker/thread settings), judge
prompt verbatim, and the cost model in `run_ladder.py`. Hardware: 96-core
CPU host, no GPU; peak memory ≈50 GB (Qdrant's 12M-point HNSW plus eight
indexer workers; estimate — the host is shared, per-process accounting was
not collected), ~110 GB disk. Version pins:
pathway 0.31.2-dev915, qdrant-client 1.18/server 1.15.5, e5-small-v2
revision as fetched 2026-07-24.
