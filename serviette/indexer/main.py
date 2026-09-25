"""Entrypoint for ``serviette indexer``.

Multi-worker runs: when ``indexer.workers`` > 1 the process prepares the
vector-DB target once, then re-executes itself through
``pathway spawn --processes N`` (worker *processes*, one thread each) — the
official Pathway mechanism, which wires up the inter-process coordination env
(``PATHWAY_PROCESS_ID`` etc.). Spawned children detect that env and skip both
the re-spawn and the (already done) backend preparation. Signals sent to the
parent are forwarded to the spawner *and* the workers (see
:func:`run_forwarding_signals`) so a stop never leaves orphans behind.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import signal
import subprocess
import sys
import time

from serviette.config.schema import require_source_dirs
from serviette.indexer.config import load_indexer_config
from serviette.indexer.graph import run_indexer

logger = logging.getLogger(__name__)


_KILL_GRACE = 15.0


def run_forwarding_signals(command: list[str], *, grace: float = _KILL_GRACE) -> int:
    """Run ``command`` to completion, forwarding SIGTERM/SIGINT to its whole
    process group; returns its exit code.

    ``pathway spawn`` starts the worker processes as plain children and
    installs no signal handlers of its own, so a SIGTERM delivered to *this*
    process (``serviette up``, systemd, ``docker stop``) used to kill just
    the Python waiting on it and leave the spawner and every worker running
    — still writing to the store, still holding the DuckDB lock and the
    inter-worker ports. The child gets its own session so a terminal Ctrl-C
    is not delivered to it twice (once by the terminal, once forwarded);
    whatever signal reaches us is sent to the child's group, which covers
    the spawner and the workers alike, and escalates to SIGKILL after
    ``grace`` seconds.
    """

    proc = subprocess.Popen(command, start_new_session=True)
    forwarded: list[int] = []

    def _forward(signum, _frame):
        forwarded.append(signum)
        try:
            os.killpg(proc.pid, signum)
        except ProcessLookupError:
            pass

    previous = {
        sig: signal.signal(sig, _forward) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    kill_at: float | None = None
    try:
        while True:
            try:
                return proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            if not forwarded:
                continue
            if kill_at is None:
                kill_at = time.monotonic() + grace
            elif time.monotonic() >= kill_at:
                logger.warning("worker processes did not stop in %.0fs; killing them", grace)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                return proc.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _spawn_workers(workers: int, argv: list[str], first_port: int = 10000) -> int:
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
        "--first-port",
        str(first_port),
        sys.executable,
        "-m",
        "serviette.cli",
        "indexer",
        *argv,
    ]
    return run_forwarding_signals(command)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="serviette indexer")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default: INFO)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper())
    config = load_indexer_config(args.config)
    # A mistyped folder must stop here with a plain message, not become an
    # empty index (the fs connector watches a missing path without complaint).
    require_source_dirs(config)

    inside_spawn = "PATHWAY_PROCESS_ID" in os.environ
    if config.indexer.workers > 1 and not inside_spawn:
        # Check the persistence fingerprint and prepare the target once, in
        # the parent, before any worker starts writing.
        from serviette.indexer.fingerprint import check_fingerprint
        from serviette.indexer.prepare import prepare_backend

        check_fingerprint(config)
        prepare_backend(config)
        code = _spawn_workers(
            config.indexer.workers,
            ["--config", args.config, "--log-level", args.log_level],
            first_port=config.indexer.spawn_first_port,
        )
        # A signal-terminated spawner reports -N; exit 128+N like a shell would.
        raise SystemExit(128 - code if code < 0 else code)

    run_indexer(config, prepare=not inside_spawn)


if __name__ == "__main__":
    main()
