"""Unit tests for the persistence configuration fingerprint."""

from __future__ import annotations

import json

import pytest

from vetosh.config.schema import load_config_dict
from vetosh.indexer.fingerprint import _FILENAME, build_fingerprint, check_fingerprint


def _config(tmp_path, chunk_size=512, embedder_model=None, enabled=True):
    return load_config_dict(
        {
            "sources": [{"type": "fs", "path": "/data"}],
            "vector_db": {"type": "duckdb", "path": str(tmp_path / "x.duckdb")},
            "embedder": {
                "type": "openai",
                "model": embedder_model,
                "api_key": "sk-SECRET",
            },
            "splitter": {"type": "token_count", "chunk_size": chunk_size},
            "persistence": {"enabled": enabled, "path": str(tmp_path / "persist")},
        }
    )


def test_first_run_writes_fingerprint(tmp_path):
    config = _config(tmp_path)
    check_fingerprint(config)
    stored = json.loads((tmp_path / "persist" / _FILENAME).read_text())
    assert stored["splitter"]["chunk_size"] == 512


def test_secrets_never_stored(tmp_path):
    fp = build_fingerprint(_config(tmp_path))
    assert "sk-SECRET" not in json.dumps(fp)


def test_unchanged_config_passes(tmp_path):
    check_fingerprint(_config(tmp_path))
    check_fingerprint(_config(tmp_path))  # no prompt, no exception


def test_changed_splitter_aborts_without_confirmation(tmp_path, monkeypatch):
    monkeypatch.delenv("VETOSH_ACCEPT_FINGERPRINT_CHANGES", raising=False)
    check_fingerprint(_config(tmp_path, chunk_size=512))
    with pytest.raises(SystemExit, match="Refusing to start"):
        check_fingerprint(_config(tmp_path, chunk_size=256))


def test_env_confirmation_accepts_and_updates(tmp_path, monkeypatch):
    check_fingerprint(_config(tmp_path, chunk_size=512))
    monkeypatch.setenv("VETOSH_ACCEPT_FINGERPRINT_CHANGES", "1")
    check_fingerprint(_config(tmp_path, chunk_size=256))
    stored = json.loads((tmp_path / "persist" / _FILENAME).read_text())
    assert stored["splitter"]["chunk_size"] == 256
    # Accepted once — the updated fingerprint now matches without the env.
    monkeypatch.delenv("VETOSH_ACCEPT_FINGERPRINT_CHANGES")
    check_fingerprint(_config(tmp_path, chunk_size=256))


def test_embedder_change_is_flagged(tmp_path, monkeypatch):
    monkeypatch.delenv("VETOSH_ACCEPT_FINGERPRINT_CHANGES", raising=False)
    check_fingerprint(_config(tmp_path, embedder_model="text-embedding-3-small"))
    with pytest.raises(SystemExit):
        check_fingerprint(_config(tmp_path, embedder_model="text-embedding-3-large"))


def test_disabled_persistence_skips_check(tmp_path):
    check_fingerprint(_config(tmp_path, enabled=False))
    assert not (tmp_path / "persist" / _FILENAME).exists()
