"""Helpers for spinning up throwaway Docker containers in integration tests.

Kept dependency-free (just the ``docker`` CLI via subprocess) so no extra Python
package is needed. Tests that use these skip automatically when Docker is not
available.
"""

from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Callable


def docker_available() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=15,
        check=False,
    )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def free_port() -> int:
    """Dynamically allocate a free TCP port (never hardcode host ports —
    anything may already be listening on a fixed localhost port)."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def free_port_range(span: int, attempts: int = 50) -> int | None:
    """Find ``span`` consecutive free ports and return the first one.

    Needed by services that advertise absolute ports to clients (e.g.
    Pinecone Local), where container ports must be published 1:1.
    """

    for _ in range(attempts):
        base = free_port()
        try:
            for port in range(base, base + span):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
        except OSError:
            continue
        return base
    return None


def run_container(
    image: str,
    *,
    ports: dict[int, int],
    env: dict[str, str],
    command: list[str] | None = None,
) -> str:
    """Start a detached container and return its id. ``ports`` maps host->container."""

    cmd = ["docker", "run", "-d", "--rm"]
    for host, cont in ports.items():
        cmd += ["-p", f"127.0.0.1:{host}:{cont}"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(image)
    if command:
        cmd += command
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker run failed: {result.stderr}")
    return result.stdout.strip()


def stop_container(container_id: str) -> None:
    subprocess.run(
        ["docker", "stop", container_id], capture_output=True, timeout=60,
        check=False,
    )


def wait_until(predicate: Callable[[], bool], *, timeout: float = 90, interval: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception as exc:  # noqa: BLE001 - readiness probes raise until ready
            last_exc = exc
        time.sleep(interval)
    raise TimeoutError(f"Service not ready within {timeout}s (last error: {last_exc})")
