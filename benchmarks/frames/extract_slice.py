"""Extract a small deterministic slice of the dump for indexing speed tests.

Takes the first N articles (>= min-chars) in dump order and writes them like
``extract_corpus.py --full`` would — same sharding, same naming — so indexing
throughput measured on the slice extrapolates to the full corpus.

    python extract_slice.py --out data/corpus-slice --limit 45000 \
        --tfds-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from extract_corpus import SNAPSHOT, slug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=45_000)
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--tfds-dir", required=True)
    args = ap.parse_args()

    import tensorflow_datasets as tfds

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = tfds.load(SNAPSHOT, split="train", data_dir=args.tfds_dir)
    written = 0
    for ex in ds.as_numpy_iterator():
        text = ex["text"].decode("utf-8")
        if len(text) < args.min_chars:
            continue
        title = ex["title"].decode("utf-8")
        shard = out / f"{written % 256:02x}"
        shard.mkdir(exist_ok=True)
        (shard / (slug(title) + ".txt")).write_text(f"{title}\n\n{text}")
        written += 1
        if written >= args.limit:
            break
    print(f"slice: {written} articles -> {out}")


if __name__ == "__main__":
    main()
