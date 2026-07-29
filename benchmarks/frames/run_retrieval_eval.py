"""Phase-0 evaluation: gold-article recall of the retrieval stack. LLM-free.

For every question, asks a running serviette server (``/api/v1/retrieve``) for
the top-k chunks and measures **article recall** — the fraction of the
question's gold Wikipedia articles (``wiki_links``) that have at least one
chunk in the context. This is the retrieval metric the FRAMES paper reports
(their BM25 setup reached 0.12–0.15), so retrieval configurations can be
ranked against the paper and against each other at zero LLM cost.

    python run_retrieval_eval.py --questions data/questions.jsonl \
        --port 8987 --k 8 --label e5small-hybrid \
        [--out results/retrieval-e5small-hybrid.jsonl]

The gold mapping relies on corpus file names being ``slug(title).txt`` (see
extract_corpus.py); a retrieved chunk counts toward an article when its
``metadata.path`` basename matches the slug of a ``wiki_links`` title.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title)[:150]


def link_title(url: str) -> str:
    """Article title of a wiki link: fragment stripped, percent-decoding applied
    (twice — some dataset links are double-encoded), underscores to spaces."""

    tail = url.rsplit("/wiki/", 1)[-1].split("#", 1)[0]
    return urllib.parse.unquote(urllib.parse.unquote(tail)).replace("_", " ").strip()


def gold_slugs(links: list[str]) -> set[str]:
    # Casefolded: corpus files carry the dump's canonical title, links the
    # (possibly differently-cased) pre-redirect name.
    return {slug(link_title(u)).casefold() for u in links}


def post_json(url: str, body: dict, timeout: int = 120, retries: int = 4) -> dict:
    """POST with retries: heavy configurations (decompose × hybrid) can push a
    transient 408/500 out of the vector store; one lost request must not kill
    an 824-question run."""

    import time

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def hit_slug(hit: dict) -> str:
    path = (hit.get("metadata") or {}).get("path", "")
    name = Path(path).name
    return (name[:-4] if name.endswith(".txt") else name).casefold()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--port", type=int, default=8987)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    questions = [json.loads(line) for line in open(args.questions)]
    url = f"http://127.0.0.1:{args.port}/api/v1/retrieve"

    def evaluate(item: dict) -> dict:
        gold = gold_slugs(item["wiki_links"])
        hits = post_json(url, {"query": item["prompt"], "k": args.k})["results"]
        got = {hit_slug(h) for h in hits}
        covered = gold & got
        return {
            "prompt": item["prompt"],
            "gold": sorted(gold),
            "covered": sorted(covered),
            "recall": len(covered) / len(gold) if gold else 1.0,
            "full_coverage": covered == gold,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        rows = list(ex.map(evaluate, questions))

    recall = statistics.mean(r["recall"] for r in rows)
    full = statistics.mean(1.0 if r["full_coverage"] else 0.0 for r in rows)
    summary = {
        "label": args.label,
        "k": args.k,
        "questions": len(rows),
        "article_recall": round(recall, 4),
        "full_coverage_rate": round(full, 4),
    }
    print(json.dumps(summary))

    out = Path(args.out or f"results/retrieval-{args.label}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
