"""Reproduce the paper's naive baseline on this corpus: article-level BM25.

The FRAMES paper's BM25-R baseline retrieved whole Wikipedia articles with the
raw question as the query (n_docs 2 and 4), reaching 0.12-0.15 article recall
over the full ~6.7M-article dump. This script runs the identical setup over
*this benchmark's* corpus, which quantifies how much of serviette's recall
advantage is the retrieval stack and how much is the smaller haystack.

Streaming two-pass BM25 (Okapi, k1=1.5, b=0.75), no index kept in memory:
pass 1 accumulates document frequencies, pass 2 scores every article against
all questions at once, keeping a top-n heap per question.

    python run_bm25_baseline.py --questions $FRAMES_DATA/questions.jsonl \
        --corpus $FRAMES_DATA/corpus-gold $FRAMES_DATA/corpus-distractors \
        --n-docs 4
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import statistics
from pathlib import Path

from run_retrieval_eval import gold_slugs

_TOKEN = re.compile(r"\w+")
K1, B = 1.5, 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def iter_articles(roots: list[Path]):
    for root in roots:
        for path in sorted(root.rglob("*.txt")):
            yield path.stem.casefold(), path.read_text(errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--n-docs", type=int, default=4)
    ap.add_argument("--out", default="results/bm25-baseline.json")
    args = ap.parse_args()

    questions = [json.loads(line) for line in open(args.questions)]
    queries = [tokenize(q["prompt"]) for q in questions]
    roots = [Path(p) for p in args.corpus]

    # Pass 1: document frequencies and lengths. Only the query vocabulary
    # matters for scoring, so df is tracked for those tokens alone — on the
    # full ~5M-article dump a whole-vocabulary df dict would not fit the
    # memory budget, and BM25 never consults it for non-query terms.
    query_vocab = {t for q in queries for t in q}
    df: dict[str, int] = {}
    doc_count = 0
    total_len = 0
    for _slug, text in iter_articles(roots):
        doc_count += 1
        tokens = tokenize(text)
        total_len += len(tokens)
        for token in set(tokens) & query_vocab:
            df[token] = df.get(token, 0) + 1
        if doc_count % 100_000 == 0:
            print(f"pass1: {doc_count} docs", flush=True)
    avg_len = total_len / doc_count
    idf = {
        t: math.log(1.0 + (doc_count - n + 0.5) / (n + 0.5)) for t, n in df.items()
    }
    print(
        f"pass1 done: {doc_count} docs, query vocab {len(query_vocab)}, "
        f"matched {len(idf)}",
        flush=True,
    )

    # Pass 2: score every article against every question; keep top-n heaps.
    query_terms = [
        {t: idf[t] for t in q if t in idf} for q in queries
    ]
    heaps: list[list[tuple[float, str]]] = [[] for _ in questions]
    scanned = 0
    for slug, text in iter_articles(roots):
        tf: dict[str, int] = {}
        for token in tokenize(text):
            tf[token] = tf.get(token, 0) + 1
        doc_len = sum(tf.values())
        norm = K1 * (1 - B + B * doc_len / avg_len)
        for qi, terms in enumerate(query_terms):
            score = 0.0
            for token, token_idf in terms.items():
                f = tf.get(token)
                if f:
                    score += token_idf * f * (K1 + 1) / (f + norm)
            if score > 0.0:
                heap = heaps[qi]
                if len(heap) < args.n_docs:
                    heapq.heappush(heap, (score, slug))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, slug))
        scanned += 1
        if scanned % 100_000 == 0:
            print(f"pass2: {scanned} docs", flush=True)

    recalls = []
    full = []
    for q, heap in zip(questions, heaps):
        gold = gold_slugs(q["wiki_links"])
        got = {slug for _score, slug in heap}
        covered = gold & got
        recalls.append(len(covered) / len(gold) if gold else 1.0)
        full.append(1.0 if covered == gold else 0.0)

    summary = {
        "setup": f"article-level BM25@{args.n_docs}, raw question as query",
        "docs": doc_count,
        "questions": len(questions),
        "article_recall": round(statistics.mean(recalls), 4),
        "full_coverage_rate": round(statistics.mean(full), 4),
    }
    print(json.dumps(summary))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
