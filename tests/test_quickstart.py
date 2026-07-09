"""Tests for the quickstart wizard config generation."""

from __future__ import annotations

import yaml

from vetosh.config.schema import VetoshConfig, load_config_dict
from vetosh.quickstart.wizard import ScriptedPrompter, Wizard, build_config, dump_yaml

BASE = {
    "license_key": "test-license-key-123",
    "sources": [{"type": "fs", "path": "/data/docs", "glob": "**/*"}],
    "vector_db_type": "pgvector",
    "pg_connection_string": "postgresql://u:p@localhost/db",
    "collection": "vetosh_embeddings",
    "embedder_type": "openai",
    "embedder_model": "text-embedding-3-small",
    "embedder_api_key": "${OPENAI_API_KEY}",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "server_host": "127.0.0.1",
    "server_port": 8000,
    "rag_enabled": False,
    "output_path": "./config.yaml",
}


def _validate(answers: dict) -> VetoshConfig:
    config_dict = build_config(answers)
    # Round-trips through YAML the way the wizard writes it.
    reparsed = yaml.safe_load(dump_yaml(config_dict))
    return load_config_dict(reparsed)


def test_universal_config_valid():
    answers = {**BASE, "config_type": "universal"}
    cfg = _validate(answers)
    cfg.for_indexer()
    cfg.for_server()
    assert cfg.sources and cfg.server is not None


def test_indexer_only_config_valid():
    answers = {**BASE, "config_type": "indexer"}
    config_dict = build_config(answers)
    cfg = _validate(answers)
    cfg.for_indexer()
    assert "server" not in config_dict
    assert "llm" not in config_dict


def test_server_only_config_valid():
    answers = {**BASE, "config_type": "server"}
    config_dict = build_config(answers)
    cfg = _validate(answers)
    cfg.for_server()
    assert "sources" not in config_dict


def test_license_key_present_in_yaml():
    answers = {**BASE, "config_type": "universal"}
    text = dump_yaml(build_config(answers))
    assert "test-license-key-123" in text
    assert yaml.safe_load(text)["pathway_license_key"] == "test-license-key-123"


def test_rag_section_emitted_when_enabled():
    answers = {
        **BASE,
        "config_type": "server",
        "rag_enabled": True,
        "llm_type": "openai",
        "llm_model": "gpt-4o-mini",
        "llm_api_key": "${OPENAI_API_KEY}",
    }
    config_dict = build_config(answers)
    assert config_dict["llm"]["model"] == "gpt-4o-mini"


def test_milvus_vector_db():
    answers = {
        **BASE,
        "config_type": "indexer",
        "vector_db_type": "milvus",
        "milvus_uri": "http://localhost:19530",
    }
    cfg = _validate(answers)
    assert cfg.vector_db.type == "milvus"


def test_duckdb_vector_db():
    answers = {
        **BASE,
        "config_type": "indexer",
        "vector_db_type": "duckdb",
        "duckdb_path": "./embeddings.duckdb",
    }
    cfg = _validate(answers)
    assert cfg.vector_db.type == "duckdb"
    assert cfg.vector_db.path == "./embeddings.duckdb"


def test_qdrant_vector_db():
    answers = {
        **BASE,
        "config_type": "indexer",
        "vector_db_type": "qdrant",
        "qdrant_host": "qdrant.example.com",
        "qdrant_api_key": "${QDRANT_API_KEY}",
    }
    cfg = _validate(answers)
    assert cfg.vector_db.type == "qdrant"
    assert cfg.vector_db.grpc_url() == "http://qdrant.example.com:6334"


def test_mongodb_vector_db():
    answers = {
        **BASE,
        "config_type": "indexer",
        "vector_db_type": "mongodb",
        "mongodb_connection_string": "mongodb+srv://u:p@cluster.example.net",
        "mongodb_database": "vetosh",
    }
    cfg = _validate(answers)
    assert cfg.vector_db.type == "mongodb"
    assert cfg.vector_db.vector_index == "vector_index"


def test_defaults_have_no_brand_name():
    """Suggested defaults must not embed the project name."""

    answers = {**BASE, "config_type": "universal"}
    # Drop explicit overrides so build_config falls back to its own defaults.
    answers.pop("collection", None)
    config = build_config(answers)
    assert config["vector_db"]["table"] == "embeddings"
    # Persistence is a silent schema default now — not emitted by the wizard.
    assert "persistence" not in config


def _feed(tokens):
    it = iter(tokens)
    return lambda _prompt: next(it)


def test_wizard_run_collects_multiple_sources():
    """Driving run() through the scripted prompter adds two sources via the loop."""

    tokens = [
        "2",                       # config type: indexer only
        "1", "/data/a",            # source 1: filesystem (glob is YAML-only now)
        "1", "/data/b",            # source 2: filesystem
        "6",                       # add another? -> Done
        "2",                       # vector db: pgvector (1 = duckdb)
        "postgresql://u:p@h/db",   # pg connection string
        "",                        # collection (default)
        "2",                       # embedder: openai (1 = local sentence_transformer)
        "", "",                    # embedder model, api key (defaults)
        "MY-KEY",                  # license key
        "",                        # output path (default)
    ]
    prompter = ScriptedPrompter(input_fn=_feed(tokens), output_fn=lambda _s: None)
    answers = Wizard(prompter=prompter).run()

    assert answers["config_type"] == "indexer"
    assert [s["path"] for s in answers["sources"]] == ["/data/a", "/data/b"]
    assert all(s["type"] == "fs" for s in answers["sources"])

    cfg = _validate(answers)
    cfg.for_indexer()
    assert len(cfg.sources) == 2


def test_wizard_run_collects_gdrive_source():
    """The sources loop can add a Google Drive source (second source type)."""

    tokens = [
        "1",                        # config type: universal
        "2",                        # source 1 type: Google Drive (index 2)
        "drive-folder-id",          # gdrive object id
        "./creds.json",             # credentials file
        "*.pdf",                    # file name pattern
        "6",                        # add another? -> Done (fs, gdrive, s3, sharepoint, pyfilesystem, done)
        "2",                        # vector db: pgvector (1 = duckdb)
        "postgresql://u:p@h/db",
        "",                         # collection default
        "1",                        # embedder: local sentence_transformer -> no questions
        "",                         # rag? No (default)
        "LIC",                      # license
        "",                         # output path
    ]
    prompter = ScriptedPrompter(input_fn=_feed(tokens), output_fn=lambda _s: None)
    answers = Wizard(prompter=prompter).run()

    assert len(answers["sources"]) == 1
    gd = answers["sources"][0]
    assert gd["type"] == "gdrive"
    assert gd["object_id"] == "drive-folder-id"
    assert gd["file_name_pattern"] == "*.pdf"

    cfg = _validate(answers)
    cfg.for_indexer()


def test_wizard_duckdb_happy_path_is_minimal(tmp_path, monkeypatch):
    """DuckDB + local embedder: no path/table/model/splitter questions at all
    (nothing exists yet, so defaults apply silently)."""

    monkeypatch.chdir(tmp_path)  # ./embeddings.duckdb must not exist
    tokens = [
        "2",            # config type: indexer only
        "1", "/data/a",  # one filesystem source
        "6",            # done
        "1",            # vector db: duckdb -> defaults, no questions
        "1",            # embedder: local sentence_transformer -> no questions
        "K",            # license key
        "",             # output path
    ]
    prompter = ScriptedPrompter(input_fn=_feed(tokens), output_fn=lambda _s: None)
    answers = Wizard(prompter=prompter).run()
    cfg = _validate(answers)
    cfg.for_indexer()
    assert cfg.vector_db.type == "duckdb"
    assert cfg.vector_db.path == "./embeddings.duckdb"
    assert cfg.embedder.type == "sentence_transformer"
