"""Unit tests for the persistence configuration fingerprint."""

from __future__ import annotations

import json

import pytest

from serviette.config.schema import load_config_dict
from serviette.indexer.fingerprint import _FILENAME, build_fingerprint, check_fingerprint


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
    monkeypatch.delenv("SERVIETTE_ACCEPT_FINGERPRINT_CHANGES", raising=False)
    check_fingerprint(_config(tmp_path, chunk_size=512))
    with pytest.raises(SystemExit, match="Refusing to start"):
        check_fingerprint(_config(tmp_path, chunk_size=256))


def test_env_confirmation_accepts_and_updates(tmp_path, monkeypatch):
    check_fingerprint(_config(tmp_path, chunk_size=512))
    monkeypatch.setenv("SERVIETTE_ACCEPT_FINGERPRINT_CHANGES", "1")
    check_fingerprint(_config(tmp_path, chunk_size=256))
    stored = json.loads((tmp_path / "persist" / _FILENAME).read_text())
    assert stored["splitter"]["chunk_size"] == 256
    # Accepted once — the updated fingerprint now matches without the env.
    monkeypatch.delenv("SERVIETTE_ACCEPT_FINGERPRINT_CHANGES")
    check_fingerprint(_config(tmp_path, chunk_size=256))


def test_embedder_change_is_flagged(tmp_path, monkeypatch):
    monkeypatch.delenv("SERVIETTE_ACCEPT_FINGERPRINT_CHANGES", raising=False)
    check_fingerprint(_config(tmp_path, embedder_model="text-embedding-3-small"))
    with pytest.raises(SystemExit):
        check_fingerprint(_config(tmp_path, embedder_model="text-embedding-3-large"))


def test_disabled_persistence_skips_check(tmp_path):
    check_fingerprint(_config(tmp_path, enabled=False))
    assert not (tmp_path / "persist" / _FILENAME).exists()


def _config_with_embedder(tmp_path, **embedder):
    return load_config_dict(
        {
            "sources": [{"type": "fs", "path": "/data"}],
            "vector_db": {"type": "duckdb", "path": str(tmp_path / "x.duckdb")},
            "embedder": {"type": "openai", "api_key": "sk-SECRET", **embedder},
            "persistence": {"enabled": True, "path": str(tmp_path / "persist")},
        }
    )


def test_fingerprint_tracks_vector_shaping_keys_but_not_secrets_or_throughput(tmp_path):
    """Every embedder key that changes the produced vectors is part of the
    identity: ``dimensions`` (openai/bedrock), ``output_dimensionality``
    (gemini), ``document_prefix`` (e5/bge), ``api_base`` (a different model
    behind the same name). Credentials and throughput knobs are not — tuning
    ``batch_size`` must not trigger a re-index prompt."""
    fp = build_fingerprint(
        _config_with_embedder(
            tmp_path,
            model="text-embedding-3-large",
            dimensions=256,
            output_dimensionality=128,
            document_prefix="passage: ",
            query_prefix="query: ",
            api_base="http://localhost:11434/v1",
            aws_secret_access_key="AWS-SECRET",
            aws_access_key_id="AKIA-SECRET",
            aws_session_token="TOKEN-SECRET",
            batch_size=64,
            capacity=8,
            retries=3,
        )
    )
    emb = fp["embedder"]
    assert emb["dimensions"] == 256
    assert emb["output_dimensionality"] == 128
    assert emb["document_prefix"] == "passage: "
    assert emb["api_base"] == "http://localhost:11434/v1"
    for key in ("query_prefix", "batch_size", "capacity", "retries"):
        assert key not in emb
    dumped = json.dumps(fp)
    for secret in ("sk-SECRET", "AWS-SECRET", "AKIA-SECRET", "TOKEN-SECRET"):
        assert secret not in dumped


def test_dimensions_change_is_flagged(tmp_path, monkeypatch):
    monkeypatch.delenv("SERVIETTE_ACCEPT_FINGERPRINT_CHANGES", raising=False)
    check_fingerprint(_config_with_embedder(tmp_path, dimensions=256))
    with pytest.raises(SystemExit, match="Refusing to start"):
        check_fingerprint(_config_with_embedder(tmp_path, dimensions=512))
