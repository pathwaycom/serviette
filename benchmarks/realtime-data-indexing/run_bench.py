#!/usr/bin/env python3
"""Run the real-time indexing benchmark and produce the memory report.

Flow: start Qdrant → start the vetosh indexer (static run over the dataset)
and the vetosh server → sample the indexer's memory (VmRSS + kernel-tracked
peak VmHWM from /proc/1/status inside the container) every few seconds until
it finishes → replay the accuracy questions through the server's /api/v1
retrieve endpoint → write a memory-over-time plot, a CSV, and a JSON summary
into ./results.

Usage:
    python run_bench.py --size 100mb [--keep-up]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
SAMPLE_INTERVAL = 3.0
# Streaming never exits; indexing is "done" when BOTH hold for this long:
# the chunk counter stopped growing AND the indexer's CPU dropped to idle.
# (Chunk commits are bursty — the embedding backlog of a large commit can
# delay the next flush arbitrarily — but during that backlog the CPU burns
# dozens of cores, so the combination is unambiguous.)
STABLE_WINDOW = 180.0
# Below this sustained CPU rate the indexer counts as idle. Must sit between
# the streaming re-scan hum (the fs connector continuously re-stats every
# file: ~5 cores per 240k files) and the embedding phase (50-90 cores).
# A false "idle" is harmless while chunks are still growing — termination
# requires BOTH conditions — so this errs high.
IDLE_CORES = 30.0


def _compose(*args: str, size: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        env={
            **__import__("os").environ,
            "BENCH_SIZE": size,
        },
        capture_output=capture,
        text=True,
        check=False,
    )


def _container_id(size: str, service: str) -> str:
    out = _compose("ps", "-q", service, size=size, capture=True).stdout.strip()
    if not out:
        raise RuntimeError(f"container for {service} not found")
    return out


def _proc_status(container: str) -> dict[str, int] | None:
    """VmRSS/VmHWM (kB) summed over ALL processes in the container.

    With ``indexer.workers`` > 1 the container holds the spawn parent plus N
    worker processes; the interesting number is their total footprint. The
    summed VmHWM is an upper bound (individual peaks need not coincide); the
    headline peak uses the sampled RSS total instead. Returns None once the
    container has exited.
    """

    proc = subprocess.run(
        ["docker", "exec", container, "sh", "-c", "cat /proc/[0-9]*/status"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    values = {"VmRSS": 0, "VmHWM": 0}
    for line in proc.stdout.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, rest = line.split(":", 1)
            values[key] += int(rest.strip().split()[0])
    return values


def _proc_cpu_seconds(container: str) -> float | None:
    """Total CPU seconds (utime+stime) over ALL processes in the container."""

    proc = subprocess.run(
        ["docker", "exec", container, "sh", "-c", "cat /proc/[0-9]*/stat"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    ticks = 0
    for line in proc.stdout.splitlines():
        fields = line.rsplit(")", 1)[-1].split()
        if len(fields) > 12:
            ticks += int(fields[11]) + int(fields[12])  # utime + stime
    return ticks / 100.0  # USER_HZ is 100 on Linux


def _embedder_description() -> str:
    import yaml

    embedder = yaml.safe_load((ROOT / "config.yaml").read_text()).get("embedder", {})
    model = embedder.get("model", "?")
    dims = embedder.get("truncate_dim")
    return f"{model} (local{f', {dims} dims' if dims else ''})"


def _wait_http(url: str, timeout: float, message: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError(message)


def _accuracy(
    base: str, questions: list[dict], k: int
) -> tuple[int, int, list[dict]]:
    hits = 0
    misses: list[dict] = []
    for q in questions:
        resp = httpx.post(
            f"{base}/api/v1/retrieve", json={"query": q["query"], "k": k}, timeout=120
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        paths = [Path(r["metadata"].get("path", "")).name for r in results]
        if q["expected_file"] in paths:
            hits += 1
        else:
            misses.append(
                {"expected": q["expected_file"], "query": q["query"][:120], "got": paths}
            )
    return hits, len(questions), misses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="100mb", help="datasets/<size> to index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--server-port", type=int, default=8300)
    parser.add_argument("--qdrant-port", type=int, default=8301)
    parser.add_argument("--keep-up", action="store_true", help="don't docker compose down")
    args = parser.parse_args()

    dataset = ROOT / "datasets" / args.size
    docs_count = len(list((dataset / "docs").glob("*.txt")))
    questions = json.loads((dataset / "questions.json").read_text())
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    import os

    os.environ["BENCH_SERVER_PORT"] = str(args.server_port)
    os.environ["BENCH_QDRANT_PORT"] = str(args.qdrant_port)

    print(f"== dataset {args.size}: {docs_count} files, {len(questions)} questions")
    # Always start from a clean slate — leftovers from an interrupted run
    # (a pre-filled Qdrant, an old indexer) would silently corrupt every
    # measurement.
    _compose("down", "-v", size=args.size)

    # Pre-warm the HF model cache on the host: containers run HF_HUB_OFFLINE.
    import yaml as _yaml

    model = _yaml.safe_load((ROOT / "config.yaml").read_text())["embedder"].get(
        "model"
    )
    if model:
        # Filesystem check first: lock-free (old runs left root-owned lock
        # files in the mounted cache), and offline containers only need the
        # snapshot to exist.
        snapshots = (
            Path.home()
            / ".cache/huggingface/hub"
            / ("models--" + model.replace("/", "--"))
            / "snapshots"
        )
        if not (snapshots.exists() and any(snapshots.iterdir())):
            from huggingface_hub import snapshot_download

            print(f"== pre-warming model cache: {model}")
            snapshot_download(model)

    _compose("up", "-d", "qdrant", size=args.size)
    _wait_http(
        f"http://127.0.0.1:{args.qdrant_port}/readyz", 120, "qdrant not ready"
    )

    started = time.monotonic()
    _compose("up", "-d", "indexer", "server", size=args.size)
    indexer = _container_id(args.size, "indexer")
    base = f"http://127.0.0.1:{args.server_port}"
    _wait_http(f"{base}/api/v1/health", 180, "server not ready")

    def _chunks() -> int:
        try:
            value = httpx.get(f"{base}/api/v1/stats", timeout=10).json().get("chunks")
            return int(value) if value is not None else 0
        except (httpx.HTTPError, ValueError):
            return 0

    # The indexer runs in its product mode — STREAMING — and never exits, so
    # "indexing finished" is detected by the chunk counter going stable: it
    # grew at least once and has not changed for STABLE_WINDOW seconds.
    # Indexing time = the moment of the last observed increase.
    # The CSV is appended live, so an interrupted run keeps its memory curve.
    csv_path = results_dir / f"{args.size}-memory.csv"
    csv_path.write_text("elapsed_s,rss_mb,hwm_mb,chunks\n")

    samples: list[tuple[float, int, int, int]] = []  # (s, rss_kb, hwm_kb, chunks)
    last_count = 0
    last_change = started
    last_busy = started
    prev_cpu: float | None = None
    prev_time = started
    cpu_seconds: float | None = None
    while True:
        status = _proc_status(indexer)
        if status is None:
            _compose("logs", "--tail", "50", "indexer", size=args.size)
            raise SystemExit("indexer container exited unexpectedly")
        now = time.monotonic()
        count = _chunks()
        if count > last_count:
            last_count = count
            last_change = now
        cpu = _proc_cpu_seconds(indexer)
        if cpu is not None:
            if prev_cpu is not None and now > prev_time:
                if (cpu - prev_cpu) / (now - prev_time) >= IDLE_CORES:
                    last_busy = now
            prev_cpu, prev_time, cpu_seconds = cpu, now, cpu
        elapsed = now - started
        sample = (elapsed, status.get("VmRSS", 0), status.get("VmHWM", 0), last_count)
        samples.append(sample)
        with csv_path.open("a") as fh:
            fh.write(
                f"{sample[0]:.1f},{sample[1] / 1024:.1f},"
                f"{sample[2] / 1024:.1f},{sample[3]}\n"
            )
        print(
            f"\r   t={elapsed:7.0f}s  rss={status.get('VmRSS', 0) / 1024:7.1f} MB  "
            f"peak={status.get('VmHWM', 0) / 1024:7.1f} MB  chunks={last_count}",
            end="",
            flush=True,
        )
        if (
            # Every document yields at least one chunk, so completion is
            # impossible before the counter reaches the document count —
            # this cleanly excludes the low-CPU initial-scan phase.
            last_count >= docs_count
            and now - last_change >= STABLE_WINDOW
            and now - last_busy >= STABLE_WINDOW
        ):
            break
        time.sleep(SAMPLE_INTERVAL)
    print()
    duration = last_change - started

    # Vector-DB footprint: on-disk storage inside the qdrant container.
    qdrant = _container_id(args.size, "qdrant")
    du = subprocess.run(
        ["docker", "exec", qdrant, "du", "-sb", "/qdrant/storage"],
        capture_output=True,
        text=True,
    )
    db_bytes = int(du.stdout.split()[0]) if du.returncode == 0 and du.stdout else None

    stats = httpx.get(f"{base}/api/v1/stats", timeout=60).json()
    hits, total, misses = _accuracy(base, questions, args.top_k)

    # Peak = highest sampled TOTAL RSS across the container's processes (the
    # summed VmHWM in the CSV is only an upper bound for multi-process runs).
    peak_mb = max(s[1] for s in samples) / 1024 if samples else 0.0
    size_mb = sum(
        f.stat().st_size for f in (dataset / "docs").glob("*.txt")
    ) / 1024 / 1024
    summary = {
        "dataset": args.size,
        "corpus_mb": round(size_mb, 1),
        "documents": docs_count,
        "chunks": stats.get("chunks"),
        "indexing_seconds": round(duration, 1),
        "throughput_mb_per_min": round(size_mb / (duration / 60), 2),
        "peak_rss_mb": round(peak_mb, 1),
        "indexer_cpu_seconds": round(cpu_seconds, 1) if cpu_seconds else None,
        "avg_parallelism": round(cpu_seconds / duration, 1) if cpu_seconds else None,
        "vector_db_disk_mb": round(db_bytes / 1024 / 1024, 1) if db_bytes else None,
        "vector_db_mb_per_corpus_mb": (
            round(db_bytes / 1024 / 1024 / size_mb, 2) if db_bytes else None
        ),
        "retrieval_accuracy": f"{hits}/{total}",
        "embedder": _embedder_description(),
        # Standard text page ≈ 2000 characters (≈ bytes for English text).
        "approx_text_pages": int(size_mb * 1024 * 1024 / 2000),
        # Which questions missed top-k, with what came back instead.
        "accuracy_misses": misses,
    }
    (results_dir / f"{args.size}-summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    minutes = [s[0] / 60 for s in samples]
    ax.plot(minutes, [s[1] / 1024 for s in samples], label="indexer RSS", linewidth=1.8)
    ax.axhline(peak_mb, linestyle="--", linewidth=1, alpha=0.6, label=f"peak {peak_mb:.0f} MB")
    ax.set_xlabel("minutes since start")
    ax.set_ylabel("memory, MB")
    ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.plot(minutes, [s[3] for s in samples], color="tab:green", alpha=0.6, linewidth=1.4, label="chunks in DB")
    ax2.set_ylabel("chunks indexed")
    ax2.set_ylim(bottom=0)
    ax.set_title(f"vetosh indexer — {args.size} corpus ({docs_count} docs, streaming)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = results_dir / f"{args.size}-memory.png"
    fig.savefig(plot_path, dpi=130)

    print(json.dumps(summary, indent=2))
    print(f"plot: {plot_path}\ncsv:  {csv_path}")

    if not args.keep_up:
        _compose("down", "-v", size=args.size)


if __name__ == "__main__":
    main()
