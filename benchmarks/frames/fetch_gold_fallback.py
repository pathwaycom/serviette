"""Fetch gold articles missing from the dump via the Wikipedia API.

The dump stores articles under their canonical titles; a ``wiki_links`` URL
pointing at a redirect therefore finds nothing in ``extract_corpus.py``. This
script resolves each missing title through the live API (following redirects)
but requests the revision **as of the paper's snapshot date** (2023-06-01), so
the corpus stays drift-free.

    python fetch_gold_fallback.py --missing data/gold-missing.txt --out data/corpus-gold
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

SNAPSHOT_TS = "2023-06-01T00:00:00Z"
UA = {"User-Agent": "serviette-frames-bench/1.0 (github.com/pathwaycom/serviette)"}


def slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title)[:150]


def fetch_at_snapshot(title: str) -> tuple[str, str] | None:
    """(resolved_title, plaintext) of the last revision before the snapshot."""

    # Step 1: resolve redirects to the canonical title.
    api = (
        "https://en.wikipedia.org/w/api.php?action=query&redirects=1&format=json"
        "&formatversion=2&titles=" + urllib.parse.quote(title)
    )
    with urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=30) as r:
        pages = json.load(r).get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        return None
    resolved = pages[0]["title"]

    # Step 2: the extract of the revision as of the snapshot date. TextExtracts
    # cannot target old revisions, so pull the wikitext of that revision and
    # strip markup crudely — good enough for retrieval text.
    api = (
        "https://en.wikipedia.org/w/api.php?action=query&prop=revisions"
        "&rvprop=content&rvslots=main&rvlimit=1&format=json&formatversion=2"
        f"&rvstart={SNAPSHOT_TS}&titles=" + urllib.parse.quote(resolved)
    )
    with urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=60) as r:
        pages = json.load(r).get("query", {}).get("pages", [])
    revisions = pages[0].get("revisions") if pages else None
    if not revisions:
        return None
    wikitext = revisions[0]["slots"]["main"]["content"]
    text = strip_wikitext(wikitext)
    return resolved, text


def strip_wikitext(src: str) -> str:
    """Rough wikitext -> plaintext: drops templates, refs, links markup."""

    text = re.sub(r"(?s)<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", src)
    text = re.sub(r"(?s)\{\{(?:[^{}]|\{\{[^{}]*\}\})*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"(?m)^[=]{2,}.*[=]{2,}$", "", text)
    text = re.sub(r"(?s)<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    titles = [t for t in Path(args.missing).read_text().splitlines() if t.strip()]
    print(f"fetching {len(titles)} missing gold articles @ {SNAPSHOT_TS}")
    failed: list[str] = []
    for i, title in enumerate(titles):
        result = None
        # Wikipedia rate-limits bursts with HTTP 429; back off patiently — this
        # script handles a few hundred stragglers, not the whole corpus.
        for attempt in range(5):
            try:
                result = fetch_at_snapshot(title)
                break
            except Exception:
                time.sleep(10 * (attempt + 1))
        if result is None:
            failed.append(title)
        else:
            resolved, text = result
            # Keep the *link* title in the filename: recall matching walks the
            # question's wiki_links, which use the pre-redirect names.
            (out / (slug(title) + ".txt")).write_text(f"{resolved}\n\n{text}")
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(titles)}", flush=True)
        time.sleep(1.5)  # be polite to the API
    if failed:
        print(f"FAILED ({len(failed)}): " + "; ".join(failed[:10]))
    print("DONE")


if __name__ == "__main__":
    main()
