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
import re
import signal
import subprocess
import sys
import time

from serviette.config.schema import (
    ServietteConfig,
    require_multi_process_backend,
    require_openai_credentials,
    require_source_dirs,
)

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.3
_TERM_GRACE = 10.0


# How many of a child's last output lines are kept for the failure report.
_TAIL_LINES = 80

# Routine lines of a child's output, hidden unless ``--verbose``: INFO/DEBUG
# records of Python logging in either common format ("INFO:name:msg",
# "INFO msg", uvicorn's "INFO:     msg"), and tqdm progress bars. Everything
# else — warnings, errors, tracebacks (whose continuation lines are
# indented), Rust panics, a child's own print() — passes through.
_ROUTINE_LINE = re.compile(r"^(INFO|DEBUG)\b|^\s*(Batches|Loading weights):\s")


def _is_routine(line: str) -> bool:
    return bool(_ROUTINE_LINE.match(line)) or not line.strip()


def _spawn(
    command: str,
    config_path: str,
    *,
    env: dict[str, str] | None = None,
    verbose: bool = True,
) -> subprocess.Popen:
    """Start a child whose output passes through ``up``.

    ``up`` keeps the child's last lines (``proc.tail``) so that, when the
    child dies, the report can name the cause instead of leaving the user to
    find the right traceback among the engine's INFO lines. With ``verbose``
    every line reaches the terminal as it arrives; otherwise only the ones
    that are not routine (see ``_ROUTINE_LINE``) — the tail keeps them all
    either way. ``PYTHONUNBUFFERED`` keeps the child from block-buffering now
    that its stdout is a pipe.
    """

    import os
    import threading
    from collections import deque

    proc = subprocess.Popen(
        [sys.executable, "-m", "serviette.cli", command, "--config", config_path],
        env={**(env if env is not None else os.environ), "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    tail: deque[str] = deque(maxlen=_TAIL_LINES)

    def forward() -> None:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            tail.append(line.rstrip("\n"))
            if verbose or not _is_routine(line):
                sys.stderr.write(line)
                sys.stderr.flush()

    threading.Thread(target=forward, name=f"up-{command}-output", daemon=True).start()
    proc.tail = tail  # type: ignore[attr-defined]
    return proc


# Known ways a child dies, matched against its last output lines: the
# cause named for the user, and what to do about it.
_FAILURE_HINTS: list[tuple[str, str, str]] = [
    (
        r"Missing credentials|OPENAI_API_KEY",
        "no OpenAI API key was available",
        "export OPENAI_API_KEY=sk-... (or set api_key in the config section) and start again.",
    ),
    (
        r"insufficient_quota|exceeded your current quota",
        "the OpenAI account has no remaining quota",
        ("check billing at https://platform.openai.com, or switch the embedder to "
        "sentence_transformer (free, local)."),
    ),
    (
        r"Error code: 429|RateLimitError|rate_limit_exceeded",
        ("OpenAI rate limit (HTTP 429) while embedding — the indexer stops at the "
        "first failed batch"),
        ("start again (persistence resumes from what was already processed); to stay "
        "under the limit index the folder in parts or lower indexer.workers."),
    ),
    (
        r"Error code: 401|AuthenticationError|Incorrect API key",
        "the OpenAI API key was rejected",
        "check OPENAI_API_KEY / api_key in the config.",
    ),
    (
        r"address already in use",
        "the server port is already taken",
        "stop the other process or change server.port.",
    ),
    (
        r"DataDirLockedError|another process holds the lock",
        "the embedded database file is locked by another process",
        "stop the other serviette/Milvus process, or use a database server.",
    ),
    (
        r"license",
        "the Pathway license key is missing or invalid",
        ("get a free key at https://pathway.com/framework/get-license and export "
        "PATHWAY_LICENSE_KEY."),
    ),
    (
        (r"Connection refused|error sending request|ConnectError|Failed to connect|"
        r"Name or service not known"),
        "a network service (the vector database or an API) could not be reached",
        "check that it is running and that host/port in the config are right.",
    ),
]


def _failure_report(name: str, code: int, tail: list[str] | None) -> str:
    """The block printed when a child dies: unmistakable, with the cause
    recognised from the child's last lines where possible."""

    import re

    cause = advice = None
    if tail:
        text = "\n".join(tail)
        for pattern, hint_cause, hint_advice in _FAILURE_HINTS:
            if re.search(pattern, text, re.IGNORECASE):
                cause, advice = hint_cause, hint_advice
                break
    lines = [
        "",
        "=" * 72,
        f"serviette up STOPPED: the {name} exited with code {code}.",
    ]
    if cause:
        lines.append(f"Cause: {cause}.")
        lines.append(f"What to do: {advice}")
    else:
        lines.append(
            f"The {name}'s last traceback above says why; the lines before it "
            "are routine engine output."
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def _report_exit(name: str, code: int, proc: subprocess.Popen) -> None:
    import time as _time

    # Give the forwarding thread a moment to drain the child's last lines.
    _time.sleep(0.2)
    logger.error(_failure_report(name, code, list(getattr(proc, "tail", None) or [])))


def _confirm_fingerprint(config: ServietteConfig) -> dict[str, str]:
    """Ask the fingerprint question here, not in the indexer child.

    The child shares this terminal; its ``input()`` prompt would sit under
    the "indexing in progress" heartbeat printed every few seconds, so the
    user sees progress while the indexer actually waits for a "yes". Same
    check (diff, prompt on a TTY, refusal otherwise), same env override —
    just asked before anything else is printed. Returns the environment for
    the child, which tells it the answer is already given.
    """

    import os

    from serviette.indexer.fingerprint import ACCEPT_ENV, check_fingerprint

    check_fingerprint(config)  # updates the stored fingerprint on "yes"
    return {**os.environ, ACCEPT_ENV: "1"}


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

    The wait is bounded by ``up.index_wait_timeout``: a non-empty folder
    whose documents all yield no text (scans without OCR, audio without its
    API key, parser failures) never produces a chunk, and the indexer keeps
    running in streaming mode — without the bound ``up`` would heartbeat
    forever. On expiry the server starts over the empty index with a
    warning; whatever becomes indexable later shows up live.
    """

    started = time.monotonic()
    last_beat = 0.0
    timeout = config.up.index_wait_timeout
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
        if timeout is not None and now - started >= timeout:
            logger.warning(
                "up: no document produced a chunk within %.0fs — starting the "
                "server over an empty index anyway. Check the indexer log "
                "above for skipped or failed documents (a parser missing its "
                "package or API key, fetch errors); documents keep being "
                "indexed live. Tune or disable this with up.index_wait_timeout.",
                now - started,
            )
            return None
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


def _probe_host(host: str) -> str:
    """The address ``up`` probes for the configured bind address.

    A wildcard bind (0.0.0.0 / ::) is not connectable as written but listens
    on loopback; a specific address or hostname is reachable only as itself
    — probing loopback for it (as ``up`` used to) never succeeds. An IPv6
    literal must be bracketed in a URL.
    """

    if host in ("", "0.0.0.0"):
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    if ":" in host:
        return f"[{host}]"
    return host


def _server_ready(config: ServietteConfig) -> bool:
    """True once the server answers ``/api/v1/health`` on its bind address."""

    import httpx

    url = (
        f"http://{_probe_host(config.server.host)}:{config.server.port}"
        "/api/v1/health"
    )
    try:
        # trust_env=False: a loopback probe must never go through the
        # http_proxy / HTTPS_PROXY of the environment — corporate proxies
        # refuse 127.0.0.1 and the wait below would never end.
        response = httpx.get(url, timeout=2.0, trust_env=False)
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


def run(config: ServietteConfig, config_path: str, *, verbose: bool = False) -> int:
    """Supervise the two children; returns the exit code for the CLI.

    ``verbose`` passes the children's full output through; by default the
    engine's routine INFO lines are hidden so that ``up``'s own progress and
    the URL to open stay visible (they are still kept for a failure report).
    """

    config.for_indexer()
    config.for_server()
    # The readiness probe goes through httpx, which logs every request at
    # INFO — that would print one line per health poll (here rather than in
    # ``main`` so that ``serviette demo``, which calls ``run`` directly, is
    # covered too).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Fail fast on a mistyped source folder: otherwise the indexer watches
    # nothing, the "empty folder" path below starts the server, and the user
    # only sees "no relevant context" answers.
    require_source_dirs(config)
    # Fail before anything starts on a backend that cannot be shared by the
    # two processes (Milvus Lite): started anyway, the indexer dies on the
    # file lock our own readiness probe holds.
    require_multi_process_backend(config)
    # The indexer would die on the SDK's "Missing credentials" traceback
    # seconds after "indexing in progress", the server on its first request.
    require_openai_credentials(config, roles=("embedder", "llm"))
    _warn_duckdb_streaming(config)

    indexer = _spawn(
        "indexer", config_path, env=_confirm_fingerprint(config), verbose=verbose
    )
    logger.info("up: indexer started (pid %d)", indexer.pid)
    if not verbose:
        logger.info(
            "up: showing progress, warnings and errors only — run with "
            "--verbose for the full indexer and server log"
        )

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
            _report_exit("indexer", failed, indexer)
            return failed
        if shutdown_requested:
            logger.info("up: shutting down")
            return 0

        server = _spawn("server", config_path, verbose=verbose)
        logger.info("up: index is ready — starting the server (pid %d)", server.pid)
        # The port is not listening until the server's warm-up finishes;
        # announcing the URL before that sends the user to a "connection
        # refused" page.
        failed = _wait_for_server(
            config, server, indexer, should_stop=lambda: shutdown_requested
        )
        if failed is not None:
            if server.poll() is not None:
                _report_exit("server", failed, server)
            else:
                _report_exit("indexer", failed, indexer)
            return failed
        if shutdown_requested:
            logger.info("up: shutting down")
            return 0
        logger.info(
            "\n\n  Ready — open %s\n  (Ctrl-C stops the indexer and the server)\n",
            _server_url(config),
        )
        static_done_logged = False
        while True:
            if shutdown_requested:
                logger.info("up: shutting down")
                return 0

            server_code = server.poll()
            if server_code is not None:
                _report_exit("server", server_code, server)
                return server_code

            indexer_code = indexer.poll()
            if indexer_code is not None and indexer_code != 0:
                _report_exit("indexer", indexer_code, indexer)
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="pass the indexer's and the server's full output through "
        "(by default routine INFO lines are hidden)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(run(load_config(args.config), args.config, verbose=args.verbose))
