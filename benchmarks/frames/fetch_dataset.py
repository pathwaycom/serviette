"""Fetch the FRAMES questions (google/frames-benchmark) into questions.jsonl.

The dataset revision is pinned (--revision, defaulting to the snapshot this
benchmark was built against) so re-runs are reproducible even if the upstream
dataset changes. Every row keeps the fields the harness needs: the prompt, the
gold answer, the reasoning type, and the gold Wikipedia article links (used
both to build the corpus and to compute article recall, the paper's retrieval
metric).

Usage:
    python fetch_dataset.py [--out DATA_DIR] [--revision SHA]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PINNED_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="data directory")
    ap.add_argument("--revision", default=PINNED_REVISION)
    args = ap.parse_args()

    url = (
        "https://huggingface.co/api/datasets/google/frames-benchmark/parquet"
        "/default/test/0.parquet"
    )
    # The parquet API serves the latest revision; verify it matches the pin so
    # a silent upstream change fails loudly instead of skewing results.
    import urllib.request

    with urllib.request.urlopen(
        "https://huggingface.co/api/datasets/google/frames-benchmark"
    ) as r:
        sha = json.load(r)["sha"]
    if sha != args.revision:
        raise SystemExit(
            f"google/frames-benchmark moved to revision {sha}, but this "
            f"benchmark pins {args.revision}. Review the diff and update "
            "PINNED_REVISION deliberately."
        )

    df = pd.read_parquet(url)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "questions.jsonl"
    n_links = 0
    with out.open("w") as f:
        for _, row in df.iterrows():
            links = [
                row[c]
                for c in df.columns
                if c.startswith("wiki")
                and isinstance(row[c], str)
                and row[c].startswith("http")
            ]
            n_links += len(links)
            f.write(
                json.dumps(
                    {
                        "prompt": row["Prompt"],
                        "answer": row["Answer"],
                        "reasoning": row.get("reasoning_types", ""),
                        "wiki_links": links,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"questions: {len(df)} | gold links: {n_links} | -> {out}")


if __name__ == "__main__":
    main()
