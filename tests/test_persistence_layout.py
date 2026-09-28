"""Layout of the working directory (``workdir_path``): serviette's own
artifacts at the top level, the Pathway engine confined to the
``persistence`` subdirectory (the engine logs an ERROR for every foreign
entry in the directory it is given), and nothing else touched."""

from __future__ import annotations

import pytest

from serviette.config.schema import load_config_dict
from serviette.indexer.fingerprint import FINGERPRINT_FILENAME, check_fingerprint
from serviette.indexer.graph import prepare_persistence_dir


def _config(tmp_path, **extra):
    return load_config_dict(
        {
            "sources": [{"type": "fs", "path": "/data"}],
            "vector_db": {"type": "duckdb", "path": str(tmp_path / "x.duckdb")},
            "embedder": {"type": "mock"},
            "workdir_path": str(tmp_path / "workdir"),
            **extra,
        }
    )


def test_engine_dir_is_a_subdirectory_of_the_workdir(tmp_path):
    config = _config(tmp_path)
    engine_dir = prepare_persistence_dir(config)
    assert engine_dir == tmp_path / "workdir" / "persistence"
    assert engine_dir.is_dir()
    assert config.persistence_dir() == engine_dir


def test_fingerprint_stays_out_of_the_engine_dir(tmp_path):
    config = _config(tmp_path)
    check_fingerprint(config)
    engine_dir = prepare_persistence_dir(config)
    assert (tmp_path / "workdir" / FINGERPRINT_FILENAME).is_file()
    assert list(engine_dir.iterdir()) == []


def test_foreign_files_in_the_workdir_are_left_alone(tmp_path):
    """Whatever else the user keeps in the working directory (a DuckDB store,
    a config, notes) is not serviette's to move — in particular never into
    the engine's directory, where it would be reported as corrupt state."""

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "store.duckdb").write_bytes(b"vectors")
    (workdir / "config.yaml").write_text("x: 1")
    config = _config(tmp_path)
    check_fingerprint(config)
    engine_dir = prepare_persistence_dir(config)
    assert (workdir / "store.duckdb").read_bytes() == b"vectors"
    assert (workdir / "config.yaml").is_file()
    assert list(engine_dir.iterdir()) == []
    assert {p.name for p in workdir.iterdir()} == {
        "store.duckdb",
        "config.yaml",
        FINGERPRINT_FILENAME,
        "persistence",
    }


def test_default_workdir_is_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config_dict(
        {
            "sources": [{"type": "fs", "path": "/data"}],
            "vector_db": {"type": "duckdb", "path": "x.duckdb"},
            "embedder": {"type": "mock"},
        }
    )
    assert config.workdir_path == "./serviette-workdir"
    assert prepare_persistence_dir(config).resolve() == (
        tmp_path / "serviette-workdir" / "persistence"
    ).resolve()


def test_persistence_path_is_rejected_with_a_pointer_to_workdir(tmp_path):
    with pytest.raises(ValueError, match="workdir_path"):
        _config(tmp_path, persistence={"enabled": True, "path": "/elsewhere"})
