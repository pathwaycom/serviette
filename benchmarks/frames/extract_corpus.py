"""Extract the benchmark corpus from the paper's exact Wikipedia snapshot.

Reads TFDS ``wikipedia/20230601.en`` — the same June 1, 2023 dump the FRAMES
paper indexed — and writes two corpora of plain-text files:

- ``corpus-gold/``: the articles referenced by the questions' ``wiki_links``
  (one file per article, named by a slug of the title so retrieval hits can be
  mapped back to gold articles for the recall metric);
- ``corpus-distractors/``: a deterministic ~N-article sample of the remaining
  dump (hash-based selection on the title, so the sample is independent of
  read order and reproducible without a stored list), sharded into
  subdirectories to keep directory sizes sane.

Run inside the TFDS venv (tensorflow-cpu + tensorflow-datasets), not the
serviette venv:

    python extract_corpus.py --questions data/questions.jsonl --out data \
        [--distractors 100000] [--min-chars 500]

Gold titles that do not match a dump page (usually redirects) are listed in
``data/gold-missing.txt``; fetch those few via ``fetch_gold_fallback.py``,
which asks the live Wikipedia API for the revision as of 2023-06-01 — same
snapshot date, so no content drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
from pathlib import Path

SNAPSHOT = "wikipedia/20230601.en"
# Selection threshold denominator: titles are hashed into [0, 1) and kept when
# below distractors/estimated_pages. The dump has ~6.7M pages; the exact count
# is discovered at runtime and the threshold adjusted in one pass.
ESTIMATED_PAGES = 6_700_000


def slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title)[:150]


def norm(title: str) -> str:
    return title.replace("_", " ").strip().casefold()


def title_of_link(url: str) -> str:
    # Fragment stripped (section anchors / text-fragments point into an
    # article, the article itself is the gold unit); double-unquote covers the
    # occasional double-encoded dataset link.
    tail = url.rsplit("/wiki/", 1)[-1].split("#", 1)[0]
    return urllib.parse.unquote(urllib.parse.unquote(tail)).replace("_", " ").strip()


def title_hash01(title: str) -> float:
    digest = hashlib.blake2b(title.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--distractors", type=int, default=100_000)
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument(
        "--full",
        action="store_true",
        help="also write every dump article >= min-chars into corpus-full/ "
        "(the star-run corpus; ~20 GB)",
    )
    ap.add_argument(
        "--tfds-dir",
        default=None,
        help="local TFDS data dir with the prepared shards (downloaded from "
        "gs://tfds-data); falls back to streaming from GCS when omitted",
    )
    args = ap.parse_args()

    import tensorflow_datasets as tfds

    out = Path(args.out)
    gold_dir = out / "corpus-gold"
    distractor_dir = out / "corpus-distractors"
    gold_dir.mkdir(parents=True, exist_ok=True)
    distractor_dir.mkdir(parents=True, exist_ok=True)
    full_dir = out / "corpus-full"
    if args.full:
        full_dir.mkdir(parents=True, exist_ok=True)

    gold_titles: dict[str, str] = {}  # normalized -> display
    for line in open(args.questions):
        for link in json.loads(line)["wiki_links"]:
            title = title_of_link(link)
            gold_titles[norm(title)] = title
    print(f"gold titles to find: {len(gold_titles)}", flush=True)

    threshold = args.distractors / ESTIMATED_PAGES
    if args.tfds_dir:
        ds = tfds.load(SNAPSHOT, split="train", data_dir=args.tfds_dir)
    else:
        ds = tfds.load(SNAPSHOT, split="train", try_gcs=True)

    found: set[str] = set()
    n_distractors = 0
    n_full = 0
    for i, ex in enumerate(ds.as_numpy_iterator()):
        title = ex["title"].decode("utf-8")
        text = ex["text"].decode("utf-8")
        key = norm(title)
        if key in gold_titles:
            (gold_dir / (slug(title) + ".txt")).write_text(f"{title}\n\n{text}")
            found.add(key)
        elif len(text) >= args.min_chars and title_hash01(title) < threshold:
            shard = distractor_dir / f"{n_distractors % 256:02x}"
            shard.mkdir(exist_ok=True)
            (shard / (slug(title) + ".txt")).write_text(f"{title}\n\n{text}")
            n_distractors += 1
        if args.full and len(text) >= args.min_chars:
            shard = full_dir / f"{n_full % 4096:03x}"
            shard.mkdir(exist_ok=True)
            (shard / (slug(title) + ".txt")).write_text(f"{title}\n\n{text}")
            n_full += 1
        if (i + 1) % 200_000 == 0:
            print(
                f"{i + 1} pages | gold {len(found)}/{len(gold_titles)} "
                f"| distractors {n_distractors}",
                flush=True,
            )

    missing = sorted(gold_titles[k] for k in gold_titles.keys() - found)
    (out / "gold-missing.txt").write_text("\n".join(missing))
    print(
        f"DONE: gold {len(found)}/{len(gold_titles)} "
        f"(missing {len(missing)} -> gold-missing.txt, likely redirects; "
        f"run fetch_gold_fallback.py) | distractors {n_distractors}"
        + (f" | full {n_full}" if args.full else ""),
        flush=True,
    )


if __name__ == "__main__":
    main()
