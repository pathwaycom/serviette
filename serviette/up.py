"""``serviette up`` — run the indexer and the server together from one config.

A deliberately dumb supervisor: it spawns ``serviette indexer``, waits until
the vector store is queryable (printing an "indexing in progress" heartbeat
— the chat page must never open onto a guaranteed error), then spawns
``serviette server`` and waits until its ``/api/v1/health`` answers before
printing the URL to open (uvicorn binds the port only after the lifespan
warm-up — a local embedder imports torch and loads its model there, tens of
seconds during which the port is not even listening). No refresh loops —
every backend runs its native streaming mode, so freshness is seconds
everywhere. Teardown rules:

- SIGINT/SIGTERM → terminate both children, exit 0.
- server exits (any code) → terminate the indexer, exit with the server code.
- indexer exits non-zero → terminate the server, exit with that code.
- indexer exits 0 (all sources were ``mode: static``) → one-shot indexing
  finished; the server keeps serving.

This is a dev/demo convenience; production deployments still run the
components separately (see docs/README "Scaling").
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import time

from serviette.config.schema import ServietteConfig, require_source_dirs

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.3
_TERM_GRACE = 10.0


def _spawn(command: str, config_path: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "serviette.cli", command, "--config", config_path]
    )


def _terminate(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_TERM_GRACE)
    except subprocess.TimeoutExpired:
        logger.warning("%s did not stop in %.0fs; killing it", name, _TERM_GRACE)
        proc.kill()
        proc.wait()


def _warn_duckdb_streaming(config: ServietteConfig) -> None:
    # Pathway builds with detach_between_batches in the duckdb sink
    # (>= 0.32.1) release the file lock between minibatches and the
    # server reads the file concurrently; only warn on older builds, where a
    # STREAMING indexer holds the file read-write for its lifetime.
    if config.vector_db and config.vector_db.type == "duckdb" and any(
        src.mode == "streaming" for src in config.sources
    ):
        import inspect

        import pathway as pw

        supported = (
            "detach_between_batches"
            in inspect.signature(pw.io.duckdb.write).parameters
        )
        if not supported:
            logger.warning(
                "DuckDB + streaming sources on this pathway build: the indexer "
                "holds the database file read-write, so server queries will "
                "fail with a lock error until the indexer stops. Upgrade "
                "pathway (duckdb detach_between_batches) or use a "
                "client-server backend for live serving."
            )


_WAIT_HEARTBEAT = 5.0


def _sources_look_empty(config: ServietteConfig) -> bool:
    """True when every source is a local folder with no matching files.

    Only decidable for ``fs`` sources; any remote source counts as
    potentially non-empty. Used to let `serviette up` serve an empty index
    immediately (a legitimate state — drop files in later and watch them
    appear) instead of waiting forever for chunks that will never come.
    """

    import fnmatch
    from pathlib import Path

    for src in config.sources:
        if src.type != "fs":
            return False
        base = Path(src.path)
        if not base.exists():
            continue
        for candidate in base.rglob("*"):
            if candidate.is_file() and fnmatch.fnmatch(
                str(candidate.relative_to(base)), src.glob.removeprefix("**/")
            ):
                return False
    return True


def _index_ready(config: ServietteConfig, *, allow_empty: bool = False) -> bool:
    """True once the vector store answers a trivial query.

    Uses the same accessor as the server, so "ready" here is exactly
    "the first server request will not fail with not-ready".
    """

    import asyncio

    from serviette.server.accessors import build_accessor
    from serviette.server.accessors.abstract import IndexNotReadyError

    async def _probe() -> bool:
        accessor = build_accessor(config.vector_db)
        try:
            stats = await accessor.stats()
            if allow_empty:
                # Sources are empty folders: a queryable-but-empty store is
                # the correct steady state; serve it and index files live as
                # they appear.
                return True
            # prepare_backend creates empty tables/collections at indexer
            # startup — "queryable" is not "has data". The indexer runs
            # forever in streaming mode, so the only start signal is actual
            # content: wait for the first chunks.
            chunks = stats.get("chunks")
            return True if chunks is None else chunks > 0
        except IndexNotReadyError:
            return False
        finally:
            await accessor.close()

    try:
        return asyncio.run(_probe())
    except Exception:  # noqa: BLE001 - backend may not even be reachable yet
        return False


def _wait_for_index(
    config: ServietteConfig,
    indexer: subprocess.Popen,
    *,
    should_stop=lambda: False,
    ready=_index_ready,
) -> int | None:
    """Block until the store is queryable; heartbeat to the console.

    Returns the indexer's exit code if it died before producing anything
    (the caller aborts), otherwise None once the index is ready. Like
    ``_wait_for_server``, ``should_stop`` lets a signal handler cut the wait
    short (returns None; the caller checks the flag) — otherwise a SIGTERM
    during a long first indexing pass would be ignored until the first
    chunks land. ``ready`` is injectable for tests.
    """

    started = time.monotonic()
    last_beat = 0.0
    allow_empty = _sources_look_empty(config)
    if allow_empty:
        logger.info(
            "up: the source folders are empty — starting with an empty "
            "index; documents dropped in later are indexed live"
        )
    while not ready(config, allow_empty=allow_empty):
        if should_stop():
            return None
        code = indexer.poll()
        if code is not None and code != 0:
            return code
        if code == 0:
            # Static run finished but the store is still not queryable —
            # nothing was produced (e.g. empty source); let the caller
            # decide, the server can legitimately serve an empty index.
            return None
        now = time.monotonic()
        if now - last_beat >= _WAIT_HEARTBEAT:
            last_beat = now
            logger.info(
                "up: indexing in progress — the chat/API server starts once "
                "the first documents are ready (%.0fs elapsed)",
                now - started,
            )
        time.sleep(1.0)
    return None


def _server_url(config: ServietteConfig) -> str:
    """The URL a browser on this machine opens (0.0.0.0 -> localhost)."""

    host = config.server.host
    if host in ("0.0.0.0", "127.0.0.1", "::"):
        host = "localhost"
    return f"http://{host}:{config.server.port}"


def _server_ready(config: ServietteConfig) -> bool:
    """True once the server answers ``/api/v1/health``.

    Probed over loopback: ``up`` and the server share a host, and a bind on
    0.0.0.0 / :: is reachable there too.
    """

    import httpx

    try:
        response = httpx.get(
            f"http://127.0.0.1:{config.server.port}/api/v1/health", timeout=2.0
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _wait_for_server(
    config: ServietteConfig,
    server: subprocess.Popen,
    indexer: subprocess.Popen,
    *,
    should_stop=lambda: False,
    ready=_server_ready,
) -> int | None:
    """Block until the server answers its health check; heartbeat meanwhile.

    Returns the exit code of a child that died first (the caller aborts
    with it), otherwise None once the server is ready. ``should_stop`` lets
    a signal handler cut the wait short (returns None; the caller checks
    the flag). ``ready`` is injectable for tests.
    """

    started = time.monotonic()
    last_beat = 0.0
    while not ready(config):
        if should_stop():
            return None
        server_code = server.poll()
        if server_code is not None:
            return server_code
        indexer_code = indexer.poll()
        if indexer_code is not None and indexer_code != 0:
            return indexer_code
        now = time.monotonic()
        if now - last_beat >= _WAIT_HEARTBEAT:
            last_beat = now
            logger.info(
                "up: server starting — loading the query embedder; the URL "
                "appears once it answers (%.0fs elapsed)",
                now - started,
            )
        time.sleep(0.5)
    return None


def run(config: ServietteConfig, config_path: str) -> int:
    """Supervise the two children; returns the exit code for the CLI."""

    config.for_indexer()
    config.for_server()
    # Fail fast on a mistyped source folder: otherwise the indexer watches
    # nothing, the "empty folder" path below starts the server, and the user
    # only sees "no relevant context" answers.
    require_source_dirs(config)
    _warn_duckdb_streaming(config)

    indexer = _spawn("indexer", config_path)
    logger.info("up: indexer started (pid %d)", indexer.pid)

    shutdown_requested = False

    def _on_signal(signum, _frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    previous = {
        sig: signal.signal(sig, _on_signal) for sig in (signal.SIGINT, signal.SIGTERM)
    }

    server: subprocess.Popen | None = None
    try:
        # The chat page must never open onto a guaranteed "index not ready"
        # error: hold the server back until the store answers a probe.
        failed = _wait_for_index(
            config, indexer, should_stop=lambda: shutdown_requested
        )
        if failed is not None:
            logger.error(
                "up: indexer exited with code %d before the index was ready", failed
            )
            return failed
        if shutdown_requested:
            logger.info("up: shutting down")
            return 0

        server = _spawn("server", config_path)
        logger.info("up: index is ready — starting the server (pid %d)", server.pid)
        # The port is not listening until the server's warm-up finishes;
        # announcing the URL before that sends the user to a "connection
        # refused" page.
        failed = _wait_for_server(
            config, server, indexer, should_stop=lambda: shutdown_requested
        )
        if failed is not None:
            if server.poll() is not None:
                logger.error("up: server exited with code %d before it was ready", failed)
            else:
                logger.error("up: indexer exited with code %d", failed)
            return failed
        if shutdown_requested:
            logger.info("up: shutting down")
            return 0
        logger.info("up: server is ready — open %s", _server_url(config))
        static_done_logged = False
        while True:
            if shutdown_requested:
                logger.info("up: shutting down")
                return 0

            server_code = server.poll()
            if server_code is not None:
                logger.error("up: server exited with code %d", server_code)
                return server_code

            indexer_code = indexer.poll()
            if indexer_code is not None and indexer_code != 0:
                logger.error("up: indexer exited with code %d", indexer_code)
                return indexer_code
            if indexer_code == 0 and not static_done_logged:
                # Static sources: one-shot indexing done; keep serving.
                logger.info("up: indexing finished; server keeps running")
                static_done_logged = True

            time.sleep(_POLL_INTERVAL)
    finally:
        _terminate(indexer, "indexer")
        if server is not None:
            _terminate(server, "server")
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv: list[str]) -> None:
    import argparse

    from serviette.config.schema import load_config

    parser = argparse.ArgumentParser(prog="serviette up", description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # The readiness probe goes through httpx, which logs every request at
    # INFO — that would print one line per health poll.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sys.exit(run(load_config(args.config), args.config))
