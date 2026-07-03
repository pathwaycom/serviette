"""Entrypoint for ``vetosh indexer``.

Multi-worker runs: when ``indexer.workers`` > 1 the process prepares the
vector-DB target once, then re-executes itself through
``pathway spawn --processes N`` (worker *processes*, one thread each) — the
official Pathway mechanism, which wires up the inter-process coordination env
(``PATHWAY_PROCESS_ID`` etc.). Spawned children detect that env and skip both
the re-spawn and the (already done) backend preparation.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

from vetosh.indexer.config import load_indexer_config
from vetosh.indexer.graph import run_indexer

logger = logging.getLogger(__name__)


def _spawn_workers(workers: int, argv: list[str]) -> int:
    logger.info("Spawning %d Pathway worker processes via `pathway spawn`", workers)
    command = [
        sys.executable,
        "-m",
        "pathway",
        "spawn",
        "--processes",
        str(workers),
        "--threads",
        "1",
        sys.executable,
        "-m",
        "vetosh.cli",
        "indexer",
        *argv,
    ]
    return subprocess.call(command)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vetosh indexer")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default: INFO)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper())
    config = load_indexer_config(args.config)

    inside_spawn = "PATHWAY_PROCESS_ID" in os.environ
    if config.indexer.workers > 1 and not inside_spawn:
        # Check the persistence fingerprint and prepare the target once, in
        # the parent, before any worker starts writing.
        from vetosh.indexer.fingerprint import check_fingerprint
        from vetosh.indexer.prepare import prepare_backend

        check_fingerprint(config)
        prepare_backend(config)
        raise SystemExit(
            _spawn_workers(
                config.indexer.workers,
                ["--config", args.config, "--log-level", args.log_level],
            )
        )

    run_indexer(config, prepare=not inside_spawn)


if __name__ == "__main__":
    main()
