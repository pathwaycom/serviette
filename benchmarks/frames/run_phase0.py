"""Phase-0 sweep: article recall for every retrieval configuration, LLM-free.

For each configuration overlay, merges it into the base config, starts a
serviette server, waits for readiness, measures gold-article recall at k via
``run_retrieval_eval`` machinery, and stops the server. One summary line per
configuration lands in ``results/phase0-summary.jsonl`` (plus a per-question
jsonl per configuration).

    python run_phase0.py --base configs/base-e5small.yaml \
        --questions $FRAMES_DATA/questions.jsonl [--k 8] [--only label,label]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from run_retrieval_eval import gold_slugs, hit_slug, post_json

# The serviette package resolves from the repo root, not from benchmarks/.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Retrieval-only overlays (no LLM anywhere). Decompose is deliberately absent:
# it needs an LLM call per question and belongs to the paid phases.
OVERLAYS: dict[str, dict] = {
    "vector": {},
    "hybrid": {"vector_db": {"hybrid": True}},
    "mmr": {"rag": {"mmr": {"candidates": 24, "diversity": 0.3}}},
    "hybrid-mmr": {
        "vector_db": {"hybrid": True},
        "rag": {"mmr": {"candidates": 24, "diversity": 0.3}},
    },
    "ce-rerank": {"reranker": {"type": "cross_encoder", "candidates": 30}},
    "hybrid-ce-rerank": {
        "vector_db": {"hybrid": True},
        "reranker": {"type": "cross_encoder", "candidates": 30},
    },
}


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def wait_ready(port: int, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/api/v1/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"server on :{port} did not become ready")


def evaluate(questions: list[dict], port: int, k: int, concurrency: int) -> list[dict]:
    url = f"http://127.0.0.1:{port}/api/v1/retrieve"

    def one(item: dict) -> dict:
        gold = gold_slugs(item["wiki_links"])
        hits = post_json(url, {"query": item["prompt"], "k": k}, timeout=300)["results"]
        covered = gold & {hit_slug(h) for h in hits}
        return {
            "prompt": item["prompt"],
            "recall": len(covered) / len(gold) if gold else 1.0,
            "full_coverage": len(covered) == len(gold),
        }

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(one, questions))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-separated overlay labels")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.base).read_text())
    questions = [json.loads(line) for line in open(args.questions)]
    labels = [s for s in args.only.split(",") if s] or list(OVERLAYS)
    port = base.get("server", {}).get("port", 8987)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "phase0-summary.jsonl"

    for label in labels:
        config = deep_merge(base, OVERLAYS[label])
        config.pop("sources", None)  # server needs no sources section
        with tempfile.NamedTemporaryFile(
            "w", suffix=f"-{label}.yaml", delete=False
        ) as f:
            yaml.safe_dump(config, f)
            config_path = f.name
        print(f"=== {label}: starting server", flush=True)
        server_log = (results_dir / f"phase0-{label}-server.log").open("w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "serviette.cli", "server", "--config", config_path],
            cwd=REPO_ROOT,
            env=dict(os.environ),
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(port)
            t0 = time.time()
            rows = evaluate(questions, port, args.k, args.concurrency)
            elapsed = time.time() - t0
        finally:
            proc.terminate()
            proc.wait(timeout=30)
        summary = {
            "label": label,
            "k": args.k,
            "questions": len(rows),
            "article_recall": round(statistics.mean(r["recall"] for r in rows), 4),
            "full_coverage_rate": round(
                statistics.mean(1.0 if r["full_coverage"] else 0.0 for r in rows), 4
            ),
            "seconds": round(elapsed),
        }
        print(json.dumps(summary), flush=True)
        with summary_path.open("a") as f:
            f.write(json.dumps(summary) + "\n")
        with (results_dir / f"phase0-{label}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
