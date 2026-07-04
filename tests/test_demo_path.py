"""CI smoke test for the demo path: quickstart -> up -> retrieve -> live add.

Drives the exact sequence a presenter runs: the wizard generates a config
(DuckDB minimal happy path), ``vetosh up`` supervises indexer + server, the
chat page and retrieval answer, and a file added while everything runs shows
up in results within seconds (streaming + DuckDB lock detach).

The only deviation from the generated config is the embedder, swapped for the
hermetic ``mock`` so CI needs no torch download; every other line is exactly
what quickstart wrote.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

from vetosh.quickstart.wizard import ScriptedPrompter, Wizard, build_config, dump_yaml

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parent.parent


def _wait(predicate, timeout: float, message: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception:  # noqa: BLE001 - readiness probing
            pass
        time.sleep(1)
    raise TimeoutError(message)


def test_demo_path_smoke(tmp_path, tcp_port, monkeypatch):
    monkeypatch.chdir(tmp_path)  # wizard defaults (./embeddings.duckdb) land here
    docs = tmp_path / "docs"
    docs.mkdir()
    alpha = "cats purr and chase mice in the sunny yard"
    (docs / "a.txt").write_text(alpha)

    # -- 1. quickstart: the minimal DuckDB happy path -------------------------
    tokens = iter(
        [
            "1",            # universal config
            "1", str(docs),  # one filesystem source
            "6",            # done
            "1",            # vector db: duckdb (silent defaults)
            "1",            # embedder: local (no further questions)
            "",             # enable /rag? -> No
            "TEST-KEY",     # license key
            "./config.yaml",
        ]
    )
    wizard = Wizard(
        prompter=ScriptedPrompter(input_fn=lambda _p: next(tokens), output_fn=lambda _s: None)
    )
    config = build_config(wizard.run())
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(dump_yaml(config))

    # -- hermetic adjustments: mock embedder (no torch in CI), test port ------
    data = yaml.safe_load(cfg_path.read_text())
    data["embedder"] = {"type": "mock"}
    data["server"]["port"] = tcp_port
    data["server"]["host"] = "127.0.0.1"
    data["pathway_license_key"] = os.environ.get("PATHWAY_LICENSE_KEY", "")
    cfg_path.write_text(yaml.safe_dump(data))

    # -- 2. one command up -----------------------------------------------------
    proc = subprocess.Popen(
        [sys.executable, "-m", "vetosh.cli", "up", "--config", str(cfg_path)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=f"{REPO_ROOT}{os.pathsep}" + os.environ.get("PYTHONPATH", "")),
    )
    base = f"http://127.0.0.1:{tcp_port}"

    def retrieved(query: str, expect: str) -> bool:
        resp = httpx.post(
            f"{base}/api/v1/retrieve", json={"query": query, "k": 2}, timeout=30
        )
        return resp.status_code == 200 and any(
            r["metadata"].get("path", "").endswith(expect)
            for r in resp.json()["results"]
        )

    try:
        _wait(
            lambda: httpx.get(f"{base}/api/v1/health", timeout=5).status_code == 200,
            timeout=300,
            message="server did not come up",
        )
        # The chat UI is on the same port.
        assert "<html" in httpx.get(base + "/", timeout=10).text.lower()

        # -- 3. retrieval over the initially indexed corpus -------------------
        # Generous: cold start imports the ML stack; CI runners are slow.
        _wait(lambda: retrieved(alpha, "a.txt"), 300, "initial document not indexed")

        # -- 4. the live moment: a file added while everything runs -----------
        beta = "dogs bark loudly at the postman every morning"
        (docs / "b.txt").write_text(beta)
        _wait(lambda: retrieved(beta, "b.txt"), 180, "live-added document not picked up")

        # -- clean teardown ----------------------------------------------------
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
