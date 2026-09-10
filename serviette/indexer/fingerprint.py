"""Configuration fingerprint guarding deterministic re-execution.

Chunking-related UDFs run with ``deterministic=True`` (see ``graph.py``), so
on a retraction Pathway *re-runs* them instead of replaying stored results.
That is only sound while they keep producing byte-identical outputs — which a
config edit or a library upgrade between restarts can silently break: a
document indexed under the old settings would be re-cut differently at
deletion time, its old chunk ids would not match, and the vector DB would
keep orphaned rows.

To catch that, the indexer stores the full risk-relevant objects (not just a
hash) in ``<persistence dir>/serviette-fingerprint.json``:

- the splitter config,
- the embedder identity: type, model, ``document_prefix`` and every extra
  key that shapes the vectors (``dimensions``, ``truncate_dim``,
  ``output_dimensionality``, ``api_base``, ...) — never credentials, and
  not the throughput knobs (batch size, capacity, retries),
- versions of the libraries whose behavior shapes the outputs
  (pathway, tiktoken, langchain_text_splitters).

On startup with persistence enabled the stored objects are compared with the
current ones. Any difference is reported as an explicit diff with a
human-readable explanation of the risk, and the indexer refuses to start
unless the user confirms — interactively on a TTY, or via
``SERVIETTE_ACCEPT_FINGERPRINT_CHANGES=1`` in non-interactive deployments.
Confirmation updates the stored fingerprint.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from serviette.config.schema import ServietteConfig

logger = logging.getLogger(__name__)

_FILENAME = "serviette-fingerprint.json"
_ACCEPT_ENV = "SERVIETTE_ACCEPT_FINGERPRINT_CHANGES"

_RISKS = {
    "splitter": (
        "Documents indexed before this change were cut with the OLD settings. "
        "When such a document is later deleted or modified, its chunks will be "
        "re-computed with the NEW settings, the chunk ids will not match, and "
        "the old vectors will stay in the vector DB as orphans. Safe options: "
        "revert the change, or re-index from scratch (drop the vector "
        "collection and the persistence directory)."
    ),
    "embedder": (
        "Vectors produced by different embedder models are not comparable: "
        "queries embedded with the new model will rank previously indexed "
        "documents nonsensically. Re-indexing from scratch is strongly "
        "recommended."
    ),
    "parser": (
        "Documents indexed before this change were parsed with the OLD "
        "routing (e.g. a different PDF parser or video prompt), so their "
        "text — and therefore their chunks and vectors — may differ from "
        "what the NEW routing would produce. Edits/deletions of such "
        "documents can leave orphaned vectors. If the affected formats "
        "matter, re-index from scratch; note that defaults also resolve "
        "from the environment (installed parser packages, API keys "
        "present), so this can change without touching the config."
    ),
    "libraries": (
        "A library upgrade can change tokenization and therefore chunk "
        "boundaries — with the same orphaned-vectors risk as a splitter "
        "config change. If unsure, re-index from scratch."
    ),
}


def _library_version(module: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(module)
    except Exception:  # noqa: BLE001 - absent/odd packaging must not break startup
        return None


# Embedder keys that never belong in the fingerprint: credentials (must not
# be written to disk), the server-side query prefix (does not shape stored
# vectors) and throughput knobs (tuning them must not prompt a re-index).
_EMBEDDER_EXCLUDED_KEYS = frozenset({
    "api_key",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "default_headers",
    "headers",
    "query_prefix",
    "batch_size",
    "capacity",
    "retries",
})


def _embedder_identity(config: ServietteConfig) -> dict[str, Any]:
    """Everything in the embedder section that shapes the produced vectors.

    Beyond ``type``/``model`` this is the open set of provider options that
    change the output — ``dimensions`` (OpenAI, Bedrock Titan),
    ``truncate_dim`` (sentence-transformers), ``output_dimensionality``
    (Gemini), ``document_prefix`` (e5/bge), ``api_base`` (a different model
    behind the same name) — so the section is taken wholesale minus the
    exclusions above, rather than by allow-list that would miss the next key.
    """
    if not config.embedder:
        return {}
    dumped = config.embedder.model_dump()
    identity = {
        k: v
        for k, v in dumped.items()
        if k not in _EMBEDDER_EXCLUDED_KEYS and v is not None and v != ""
    }
    # Always present, even when unset, so old and new fingerprints line up.
    identity.setdefault("type", dumped.get("type"))
    identity.setdefault("model", dumped.get("model"))
    return identity


def build_fingerprint(config: ServietteConfig) -> dict[str, Any]:
    from serviette.indexer.graph import ParserRegistry

    return {
        "splitter": config.splitter.model_dump(),
        "parser": ParserRegistry(config.parser).resolved_rules(),
        "embedder": _embedder_identity(config),
        "libraries": {
            name: _library_version(name)
            for name in ("pathway", "tiktoken", "langchain_text_splitters")
        },
    }


def _diff(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for section, risk in _RISKS.items():
        old, new = stored.get(section), current.get(section)
        if old == new:
            continue
        lines.append(f"[{section}] stored: {json.dumps(old, sort_keys=True)}")
        lines.append(f"[{section}] current: {json.dumps(new, sort_keys=True)}")
        lines.append(f"[{section}] risk: {risk}")
    return lines


def _confirm(diff_lines: list[str]) -> bool:
    if os.environ.get(_ACCEPT_ENV) == "1":
        logger.warning(
            "Fingerprint changes accepted via %s=1; proceeding.", _ACCEPT_ENV
        )
        return True
    if sys.stdin.isatty():
        answer = input(
            "The indexing configuration changed since the last run (see above). "
            "Proceed anyway, accepting these risks? Type 'yes' to continue: "
        )
        return answer.strip().lower() == "yes"
    return False


def check_fingerprint(config: ServietteConfig) -> None:
    """Verify (and maintain) the persisted fingerprint; may abort startup."""

    if not config.persistence.enabled:
        return  # no persisted state to be inconsistent with
    directory = Path(config.persistence.path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _FILENAME

    current = build_fingerprint(config)
    if not path.exists():
        path.write_text(json.dumps(current, indent=2, sort_keys=True))
        return

    try:
        stored = json.loads(path.read_text())
    except (OSError, ValueError):
        logger.warning("Unreadable fingerprint file %s; rewriting it.", path)
        path.write_text(json.dumps(current, indent=2, sort_keys=True))
        return

    diff_lines = _diff(stored, current)
    if not diff_lines:
        return

    logger.critical(
        "The indexing configuration differs from the one this persistence "
        "directory (%s) was built with:\n%s",
        directory,
        "\n".join("  " + line for line in diff_lines),
    )
    if not _confirm(diff_lines):
        raise SystemExit(
            "Refusing to start: the configuration change above can corrupt "
            "incremental updates of already-indexed documents. Either revert "
            "the change, re-index from scratch (drop the vector collection "
            f"and {directory}), or set {_ACCEPT_ENV}=1 / confirm interactively "
            "to accept the risks."
        )
    path.write_text(json.dumps(current, indent=2, sort_keys=True))
    logger.warning("Fingerprint updated to the current configuration.")
