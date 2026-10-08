"""Tests for the wizard config generation."""

from __future__ import annotations

import pytest
import yaml

from serviette import wizard
from serviette.config.schema import ServietteConfig, load_config_dict
from serviette.wizard import ScriptedPrompter, Wizard, build_config, dump_yaml


@pytest.fixture(autouse=True)
def _every_embedder_package_installed(monkeypatch):
    """The scripted runs below pick embedders by index; whether their optional
    packages happen to be installed in the test venv must not add a prompt."""
    monkeypatch.setattr(wizard, "_installed", lambda module: True)

BASE = {
    "license_key": "test-license-key-123",
    "sources": [{"type": "fs", "path": "/data/docs", "glob": "**/*"}],
    "vector_db_type": "pgvector",
    "pg_connection_string": "postgresql://u:p@localhost/db",
    "collection": "serviette_embeddings",
    "embedder_type": "openai",
    "embedder_model": "text-embedding-3-small",
    "embedder_api_key": "${OPENAI_API_KEY}",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "server_host": "127.0.0.1",
    "server_port": 8000,
    "llm_type": "mock",
    "output_path": "./config.yaml",
}


def _validate(answers: dict) -> ServietteConfig:
    config_dict = build_config(answers)
    # Round-trips through YAML the way the wizard writes it.
    reparsed = yaml.safe_load(dump_yaml(config_dict))
    # Shape only: the ${ENV} references are for the machine the config runs on.
    return load_config_dict(reparsed, strict_env=False)


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


def test_server_config_always_has_llm_mock_by_default():
    """/rag is always on: a server config carries an ``llm`` section even when
    the user picked no LLM (mock answers quote the best snippet)."""
    answers = {**BASE, "config_type": "server"}
    config_dict = build_config(answers)
    assert config_dict["llm"] == {"type": "mock"}
    _validate(answers).for_server()


def test_llm_section_openai():
    answers = {
        **BASE,
        "config_type": "server",
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
        "mongodb_database": "serviette",
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
    assert "workdir_path" not in config


def _feed(tokens):
    it = iter(tokens)
    return lambda _prompt: next(it)


def test_wizard_run_collects_multiple_sources():
    """Driving run() through the scripted prompter adds two sources via the loop."""

    tokens = [
        "2",                       # config type: indexer only
        "1", "/data/a", "3",       # source 1: filesystem (glob is YAML-only now);
                                   # the folder does not exist -> keep as typed
        "1", "/data/b", "3",       # source 2: filesystem
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
        "1",                        # embedder: local sentence_transformer
        "1",                        # embedding model: all-MiniLM-L6-v2 (English)
        "",                         # LLM for chat answers: mock (default)
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


def _minimal_answers(tokens_after_embedder: list[str], monkeypatch, tmp_path, shown=None, *, model="1"):
    """Run the wizard through the DuckDB + local-embedder path; the caller
    supplies the answers from the license question on. ``model`` is the
    answer to the embedding-model question (default: the English model);
    None omits it, for runs where another prompt follows the embedder."""

    monkeypatch.chdir(tmp_path)
    tokens = ["2", "1", "/data/a", "3", "6", "1", "1", *([model] if model else []), *tokens_after_embedder]
    prompter = ScriptedPrompter(
        input_fn=_feed(tokens), output_fn=shown.append if shown is not None else (lambda _s: None)
    )
    return Wizard(prompter=prompter).run()


def test_wizard_references_an_exported_license_key(tmp_path, monkeypatch):
    """With PATHWAY_LICENSE_KEY exported, Enter keeps the key out of the file
    and references the variable; a pasted key still wins. Without it the
    question stays required."""

    monkeypatch.setenv("PATHWAY_LICENSE_KEY", "exported-key")
    answers = _minimal_answers(["", ""], monkeypatch, tmp_path)  # Enter on the key, default output path
    assert answers["license_key"] == "${PATHWAY_LICENSE_KEY}"
    assert "exported-key" not in dump_yaml(build_config(answers))

    answers = _minimal_answers(["pasted", ""], monkeypatch, tmp_path)
    assert answers["license_key"] == "pasted"

    monkeypatch.delenv("PATHWAY_LICENSE_KEY")
    shown: list[str] = []
    answers = _minimal_answers(["", "typed", ""], monkeypatch, tmp_path, shown)
    assert answers["license_key"] == "typed"
    assert any("required" in line for line in shown)


def test_wizard_duckdb_happy_path_is_minimal(tmp_path, monkeypatch):
    """DuckDB + local embedder: no path/table/model/splitter questions at all
    (nothing exists yet, so defaults apply silently)."""

    monkeypatch.chdir(tmp_path)  # ./embeddings.duckdb must not exist
    tokens = [
        "2",            # config type: indexer only
        "1", "/data/a", "3",  # one filesystem source (missing folder: keep as typed)
        "6",            # done
        "1",            # vector db: duckdb -> defaults, no questions
        "1",            # embedder: local sentence_transformer
        "1",            # embedding model: default
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


def test_wizard_offers_to_create_a_missing_folder(tmp_path, monkeypatch):
    """A folder that does not exist is flagged and, on request, created;
    a folder that cannot be created sends the user back to the question."""

    monkeypatch.chdir(tmp_path)
    shown: list[str] = []
    new_dir = tmp_path / "fresh" / "docs"
    tokens = [
        "2",                            # indexer only
        "1", str(new_dir), "1",         # fs source: missing -> create it now
        "6", "1", "1", "1", "K", "",    # done, duckdb, local embedder, its model, license, output
    ]
    prompter = ScriptedPrompter(input_fn=_feed(tokens), output_fn=shown.append)
    answers = Wizard(prompter=prompter).run()
    assert new_dir.is_dir()
    assert answers["sources"][0]["path"] == str(new_dir)
    assert any("does not exist" in line for line in shown)
    assert any("Created" in line for line in shown)

    # Not creatable (a file is in the way): back to the question, then "keep as typed".
    (tmp_path / "file").write_text("x")
    blocked = tmp_path / "file" / "docs"
    tokens = ["2", "1", str(blocked), "1", "3", "6", "1", "1", "1", "K", ""]
    shown.clear()
    prompter = ScriptedPrompter(input_fn=_feed(tokens), output_fn=shown.append)
    answers = Wizard(prompter=prompter).run()
    assert answers["sources"][0]["path"] == str(blocked)
    assert any("Could not create" in line for line in shown)


def test_wizard_fs_source_expands_tilde_and_reasks_missing_dir(tmp_path, monkeypatch):
    """A mistyped folder is caught at the keyboard: the wizard re-asks unless
    the user explicitly keeps it; ``~`` is expanded in the answer."""

    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "docs").mkdir()
    tokens = [
        "2",                    # config type: indexer only
        "1", "/no/such/dir",    # fs source: missing folder
        "2",                    #   what now? -> enter a different path
        "~/docs",               #   an existing folder, via ~
        "6",                    # done
        "1",                    # duckdb
        "1",                    # local embedder
        "1",                    # embedding model: default
        "K",                    # license key
        "",                     # output path
    ]
    prompter = ScriptedPrompter(input_fn=_feed(tokens), output_fn=lambda _s: None)
    answers = Wizard(prompter=prompter).run()
    assert answers["sources"][0]["path"] == str(tmp_path / "docs")


def test_wizard_warns_when_the_embedder_package_is_missing(tmp_path, monkeypatch):
    """Picking the local embedder without sentence-transformers installed
    shows the install command and lets the user keep the choice."""

    monkeypatch.setattr(wizard, "_installed", lambda module: module != "sentence_transformers")
    shown: list[str] = []
    # "1" = keep it, "1" = default model, then the license key and the output path.
    answers = _minimal_answers(["1", "1", "LIC", ""], monkeypatch, tmp_path, shown, model=None)
    assert answers["embedder_type"] == "sentence_transformer"
    warning = "\n".join(shown)
    assert "sentence_transformers package" in warning
    assert 'pip install "serviette[local]" --extra-index-url https://download.pytorch.org/whl/cpu' in warning


def test_wizard_lets_the_user_switch_embedder_when_the_package_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard, "_installed", lambda module: module != "sentence_transformers")
    # "2" = choose another embedder -> openai (index 2), then its model/key defaults.
    answers = _minimal_answers(["2", "2", "", "", "LIC", ""], monkeypatch, tmp_path, model=None)
    assert answers["embedder_type"] == "openai"


def test_wizard_offers_a_multilingual_model_and_sets_its_prefixes(tmp_path, monkeypatch):
    """The default local model is English-only; picking multilingual-e5
    must also write the query/passage prefixes it was trained with."""

    answers = _minimal_answers(["LIC", ""], monkeypatch, tmp_path, model="2")
    assert answers["embedder_model"] == "intfloat/multilingual-e5-small"
    cfg = build_config(answers)
    assert cfg["embedder"] == {
        "type": "sentence_transformer",
        "model": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    }
    _validate(answers)


def test_wizard_accepts_a_custom_model_id(tmp_path, monkeypatch):
    # "4" = Other, then the id; no prefixes are assumed for an unknown model.
    answers = _minimal_answers(["BAAI/bge-small-en-v1.5", "LIC", ""], monkeypatch, tmp_path, model="4")
    cfg = build_config(answers)
    assert cfg["embedder"] == {"type": "sentence_transformer", "model": "BAAI/bge-small-en-v1.5"}
