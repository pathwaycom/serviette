"""``run_forwarding_signals``: a SIGTERM to the multi-worker indexer parent
must reach the spawner *and* its workers (no orphans)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Stands in for `pathway spawn`: starts a "worker" grandchild, records both
# pids, then waits — exactly the shape that used to leave orphans.
_SPAWNER = """
import subprocess, sys, time, os
worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
open(sys.argv[1], "w").write(f"{os.getpid()} {worker.pid}")
worker.wait()
"""

_PARENT = """
import sys
from serviette.indexer.main import run_forwarding_signals
code = run_forwarding_signals([sys.executable, "-c", {spawner!r}, sys.argv[1]])
sys.exit(code if code >= 0 else 128 - code)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


def test_sigterm_to_parent_stops_spawner_and_workers(tmp_path):
    pids_file = tmp_path / "pids"
    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT.format(spawner=_SPAWNER), str(pids_file)],
        cwd=REPO_ROOT,
    )
    try:
        deadline = time.monotonic() + 20
        while not pids_file.exists() or not pids_file.read_text().strip():
            assert time.monotonic() < deadline, "spawner did not start"
            assert parent.poll() is None, "parent exited early"
            time.sleep(0.1)
        spawner_pid, worker_pid = map(int, pids_file.read_text().split())
        assert _alive(spawner_pid) and _alive(worker_pid)

        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=30) != 0
        assert _wait_gone(spawner_pid, 10), "spawner orphaned"
        assert _wait_gone(worker_pid, 10), "worker orphaned"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        for line in [pids_file.read_text()] if pids_file.exists() else []:
            for pid in map(int, line.split()):
                if _alive(pid):
                    os.kill(pid, signal.SIGKILL)


def test_child_exit_code_is_returned():
    from serviette.indexer.main import run_forwarding_signals

    assert run_forwarding_signals([sys.executable, "-c", "raise SystemExit(7)"]) == 7
    assert run_forwarding_signals([sys.executable, "-c", "pass"]) == 0


def test_sigkill_escalation_when_child_ignores_sigterm():
    """A child that ignores SIGTERM is killed after the grace period instead
    of hanging the parent forever."""
    import threading

    from serviette.indexer.main import run_forwarding_signals

    result: list[int] = []
    stubborn = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"

    def run():
        result.append(run_forwarding_signals([sys.executable, "-c", stubborn], grace=1.0))

    # Signal handlers must be installed from the main thread; so run the
    # waiter there and fire the signal from a helper thread.
    threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    started = time.monotonic()
    run()
    assert time.monotonic() - started < 15
    assert result == [-signal.SIGKILL]
