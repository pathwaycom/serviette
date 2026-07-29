"""Phase-1/2 sweep: end-to-end accuracy for selected configurations.

Same orchestration as ``run_phase0`` (server per configuration overlay), but
the server gets an ``llm`` section and every question runs through ``/rag``
with the Appendix-B judge (``run_bench.py``).

    OPENAI_API_KEY=... python run_phase1.py --base configs/base-e5small.yaml \
        --questions $FRAMES_DATA/questions.jsonl \
        --generator gpt-4o-mini [--k 8] [--only label,label] [--limit 0]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from run_phase0 import REPO_ROOT, deep_merge, wait_ready

# End-to-end overlays: phase-0 winners plus the LLM-dependent strategies
# (decompose, adaptive) that retrieval-only ranking cannot measure.
OVERLAYS: dict[str, dict] = {
    "vector": {},
    "hybrid": {"vector_db": {"hybrid": True}},
    "hybrid-mmr": {
        "vector_db": {"hybrid": True},
        "rag": {"mmr": {"candidates": 24, "diversity": 0.3}},
    },
    "hybrid-decompose": {
        "vector_db": {"hybrid": True},
        "rag": {"decompose": {"max_subqueries": 4}},
    },
    "hybrid-decompose-adaptive": {
        "vector_db": {"hybrid": True},
        "rag": {
            "decompose": {"max_subqueries": 4},
            "adaptive": {"factor": 2, "max_iterations": 3},
        },
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--only", default="", help="comma-separated overlay labels")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.base).read_text())
    labels = [s for s in args.only.split(",") if s] or list(OVERLAYS)
    port = base.get("server", {}).get("port", 8987)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    llm_section = {
        "llm": {
            "type": "openai",
            "model": args.generator,
            "api_key": "${OPENAI_API_KEY}",
        }
    }
    # Reasoning models (gpt-5*, o*) accept only the default temperature; for
    # the rest, pin 0 for reproducibility (documented in the README).
    if not args.generator.startswith(("gpt-5", "o")):
        llm_section["llm"]["temperature"] = 0

    for label in labels:
        config = deep_merge(deep_merge(base, OVERLAYS[label]), llm_section)
        config.pop("sources", None)
        with tempfile.NamedTemporaryFile(
            "w", suffix=f"-{label}.yaml", delete=False
        ) as f:
            yaml.safe_dump(config, f)
            config_path = f.name
        run_label = f"{label}-{args.generator}"
        print(f"=== {run_label}: starting server", flush=True)
        server_log = (results_dir / f"bench-{run_label}-server.log").open("w")
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
            bench = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "run_bench.py"),
                    "--questions", args.questions,
                    "--port", str(port),
                    "--k", str(args.k),
                    "--label", run_label,
                    "--limit", str(args.limit),
                    "--concurrency", str(args.concurrency),
                    "--out", str(results_dir / f"bench-{run_label}.jsonl"),
                ],
                cwd=Path(__file__).parent,
                env=dict(os.environ),
                capture_output=True,
                text=True,
            )
            print(bench.stdout.strip(), flush=True)
            if bench.returncode != 0:
                print(f"BENCH FAILED for {run_label}:\n{bench.stderr[-2000:]}", flush=True)
            print(f"({run_label}: {time.time() - t0:.0f}s total)", flush=True)
        finally:
            proc.terminate()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
