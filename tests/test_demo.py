"""Unit tests for the ``serviette demo`` scaffolding (no processes spawned)."""

from __future__ import annotations

from pathlib import Path

from serviette.config.schema import load_config_dict
from serviette.demo import build_demo_config, choose_embedder, prepare_demo_dir


def test_demo_config_validates(tmp_path):
    config = build_demo_config(
        tmp_path, port=8123, embedder="mock", license_key="KEY"
    )
    cfg = load_config_dict(config)
    cfg.for_indexer()
    cfg.for_server()
    assert cfg.vector_db.type == "duckdb"
    assert cfg.server.port == 8123
    assert cfg.llm is not None  # chat always answers something


def test_demo_openai_key_upgrades_llm_only(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    config = build_demo_config(
        tmp_path,
        port=8000,
        embedder="sentence_transformer",
        license_key="${PATHWAY_LICENSE_KEY}",
    )
    assert config["llm"]["type"] == "openai"
    assert config["llm"]["api_key"] == "${OPENAI_API_KEY}"
    assert config["embedder"]["type"] == "sentence_transformer"
    assert "api_key" not in config["embedder"]


def test_choose_embedder_stays_local_even_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert choose_embedder() == "sentence_transformer"


def test_mock_notice_names_both_remedies(monkeypatch):
    from serviette.demo import mock_embedder_notice

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    notice = mock_embedder_notice()
    assert "MOCK EMBEDDER" in notice
    assert 'serviette[local]' in notice
    assert "--embedder openai" in notice
    assert "OPENAI_API_KEY is set" not in notice

    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert "never spends it" in mock_embedder_notice()


def test_demo_openai_embedder_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    config = build_demo_config(
        tmp_path, port=8000, embedder="openai", license_key="K"
    )
    assert config["embedder"] == {"type": "openai", "api_key": "${OPENAI_API_KEY}"}
    load_config_dict(config).for_indexer()


def test_demo_openai_embedder_requires_key(tmp_path, monkeypatch):
    import pytest

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        prepare_demo_dir(
            tmp_path, port=8000, license_key="K", embedder_choice="openai"
        )


def test_demo_embedder_switch_resets_index(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prepare_demo_dir(tmp_path, port=1, license_key="K")  # local embedder
    (tmp_path / "embeddings.duckdb").write_bytes(b"local vectors")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    cfg = prepare_demo_dir(
        tmp_path, port=1, license_key="K", embedder_choice="openai"
    )
    import yaml

    assert yaml.safe_load(cfg.read_text())["embedder"]["type"] == "openai"
    # Vectors from different embedders are not comparable: index dropped.
    assert not (tmp_path / "embeddings.duckdb").exists()


def test_prepare_demo_dir_copies_corpus_and_writes_config(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_path = prepare_demo_dir(tmp_path, port=8000, license_key="KEY")
    docs = list((tmp_path / "docs").glob("*.md"))
    assert len(docs) >= 5
    assert any(p.name == "pricing.md" for p in docs)
    assert config_path.exists()
    # The generated file round-trips through the normal config loader.
    from serviette.config.schema import load_config

    cfg = load_config(config_path)
    assert Path(cfg.sources[0].path) == tmp_path / "docs"
    # Idempotent: a second run must not fail on the existing docs dir.
    prepare_demo_dir(tmp_path, port=8000, license_key="KEY")


def test_demo_key_change_keeps_local_index(tmp_path, monkeypatch):
    """Exporting OPENAI_API_KEY between runs upgrades the LLM but must not
    touch the locally-embedded index."""
    from serviette.demo import prepare_demo_dir

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prepare_demo_dir(tmp_path, port=1, license_key="K")  # local embedder
    (tmp_path / "embeddings.duckdb").write_bytes(b"old vectors")
    (tmp_path / "persistence").mkdir()
    (tmp_path / "persistence" / "snap").write_text("x")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    cfg = prepare_demo_dir(tmp_path, port=1, license_key="K")

    import yaml

    data = yaml.safe_load(cfg.read_text())
    # The key upgrades only the LLM; the embedder — and the index — stay local.
    assert data["embedder"]["type"] == "sentence_transformer"
    assert data["llm"]["type"] == "openai"
    assert (tmp_path / "embeddings.duckdb").exists()


def test_demo_same_embedder_keeps_index(tmp_path, monkeypatch):
    from serviette.demo import prepare_demo_dir

    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    prepare_demo_dir(tmp_path, port=1, license_key="K")
    (tmp_path / "embeddings.duckdb").write_bytes(b"vectors")
    prepare_demo_dir(tmp_path, port=1, license_key="K")
    assert (tmp_path / "embeddings.duckdb").read_bytes() == b"vectors"
