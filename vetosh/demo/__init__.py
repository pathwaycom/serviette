"""``vetosh demo`` — zero-to-chat in one command.

Copies a small bundled corpus (a fictional company, Lumina Coffee Systems)
into a working directory, generates a DuckDB-backed config, and runs the
``vetosh up`` supervisor over it. The printed script walks the presenter
through the money shot: edit ``docs/pricing.md`` and watch the answer change.

Embedder/LLM selection (no questions asked). The embedder is always local —
``sentence-transformers`` when installed, the mock embedder otherwise —
so retrieval is free and the index never depends on an API key. The LLM
uses ``OPENAI_API_KEY`` when it is set (real generated answers) and falls
back to the mock LLM (the answer quotes the retrieved snippet) when not.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CORPUS_DIR = Path(__file__).parent / "corpus"

_SCRIPT = """
────────────────────────────────────────────────────────────────────────
 vetosh demo — Lumina Coffee Systems
 Chat UI:  http://localhost:{port}
           (opens after the first indexing pass — wait for the
            "index is ready — server started" line below; the first
            run also downloads the embedding model, ~1-2 min)

 Your documents live in:
   {docs_dir}
 Anything you do there — edit a file, drop in new documents (PDF, DOCX,
 scans, …), delete one — is reflected in the answers within seconds.

 A scripted moment to try first:
   1. Ask in the chat: How much does the Team tier cost?   (→ 129 EUR)
   2. Open {docs_dir}/pricing.md and change 129 EUR → 199 EUR
   3. Ask again — the answer follows the file
      (watch the “indexed … ago” counter in the header)

 Ctrl-C stops everything.
────────────────────────────────────────────────────────────────────────
"""


def choose_embedder() -> str:
    """Pick the local embedder: sentence-transformers if installed, else mock.

    Deliberately never an API embedder — the demo index must not depend on
    (or spend) anyone's API key; the key only upgrades the LLM.
    """

    if importlib.util.find_spec("sentence_transformers") is not None:
        return "sentence_transformer"
    return "mock"


def build_demo_config(
    demo_dir: Path, *, port: int, embedder: str, license_key: str
) -> dict[str, Any]:
    """Assemble the demo config dict (pure; unit-tested)."""

    config: dict[str, Any] = {
        "pathway_license_key": license_key,
        "sources": [
            {"type": "fs", "path": str(demo_dir / "docs"), "glob": "**/*"}
        ],
        "vector_db": {
            "type": "duckdb",
            "path": str(demo_dir / "embeddings.duckdb"),
            "table": "demo_embeddings",
        },
        "persistence": {
            "enabled": True,
            "backend": "filesystem",
            "path": str(demo_dir / "persistence"),
        },
        "server": {"host": "127.0.0.1", "port": port},
    }
    config["embedder"] = (
        {"type": "sentence_transformer"}
        if embedder == "sentence_transformer"
        else {"type": "mock"}
    )
    # The LLM (not the embedder) upgrades to OpenAI when a key is present.
    if os.environ.get("OPENAI_API_KEY"):
        config["llm"] = {"type": "openai", "api_key": "${OPENAI_API_KEY}"}
    else:
        config["llm"] = {"type": "mock"}
    return config


def _resolve_license_key() -> str:
    if os.environ.get("PATHWAY_LICENSE_KEY"):
        return "${PATHWAY_LICENSE_KEY}"
    if not sys.stdin.isatty():
        raise SystemExit(
            "vetosh demo needs a Pathway license key (free, one click at "
            "https://pathway.com/framework/get-license). Set "
            "PATHWAY_LICENSE_KEY and re-run."
        )
    print(
        "vetosh demo needs a (free) Pathway license key — one click at\n"
        "https://pathway.com/framework/get-license"
    )
    key = input("Paste your Pathway license key: ").strip()
    if not key:
        print("A license key is required.", file=sys.stderr)
        raise SystemExit(1)
    return key


def _reset_index_if_embedder_changed(demo_dir: Path, embedder: str) -> None:
    """Demo data is disposable: on an embedder switch (e.g. the user exported
    OPENAI_API_KEY since the last run), silently drop the old index instead
    of tripping the fingerprint guard — vectors from different embedders are
    not comparable, and re-indexing the tiny corpus takes seconds."""

    config_path = demo_dir / "config.yaml"
    if not config_path.exists():
        return
    try:
        previous = yaml.safe_load(config_path.read_text())["embedder"]["type"]
    except Exception:  # noqa: BLE001 - a hand-edited config must not crash the demo
        return
    current = embedder if embedder == "sentence_transformer" else "mock"
    if previous == current:
        return
    logger.info(
        "demo: embedder changed (%s -> %s); resetting the demo index",
        previous,
        current,
    )
    (demo_dir / "embeddings.duckdb").unlink(missing_ok=True)
    shutil.rmtree(demo_dir / "persistence", ignore_errors=True)


def prepare_demo_dir(demo_dir: Path, *, port: int, license_key: str) -> Path:
    """Copy the corpus and write config.yaml; returns the config path."""

    docs = demo_dir / "docs"
    if not docs.exists():
        shutil.copytree(_CORPUS_DIR, docs)
    embedder = choose_embedder()
    _reset_index_if_embedder_changed(demo_dir, embedder)
    if embedder == "mock":
        logger.warning(
            "Neither OPENAI_API_KEY nor sentence-transformers found: using the "
            "mock embedder (retrieval quality is NOT representative). "
            'Install "vetosh[local]" or set OPENAI_API_KEY for a real demo.'
        )
    elif embedder == "sentence_transformer":
        logger.info(
            "Using local sentence-transformers embeddings; /rag will quote "
            "retrieved snippets (set OPENAI_API_KEY for generated answers)."
        )
    config = build_demo_config(
        demo_dir, port=port, embedder=embedder, license_key=license_key
    )
    config_path = demo_dir / "config.yaml"
    config_path.write_text(
        "# generated by `vetosh demo`\n"
        + yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
    )
    return config_path


def main(argv: list[str]) -> None:
    from vetosh.config.schema import load_config
    from vetosh.up import run as up_run

    parser = argparse.ArgumentParser(prog="vetosh demo", description=__doc__)
    parser.add_argument("--dir", default="./vetosh-demo", help="Demo working directory")
    parser.add_argument("--port", type=int, default=8989)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    demo_dir = Path(args.dir).resolve()
    demo_dir.mkdir(parents=True, exist_ok=True)
    config_path = prepare_demo_dir(
        demo_dir, port=args.port, license_key=_resolve_license_key()
    )
    print(_SCRIPT.format(port=args.port, docs_dir=demo_dir / "docs"))
    sys.exit(up_run(load_config(config_path), str(config_path)))
