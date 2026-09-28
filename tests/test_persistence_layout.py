"""Layout of ``persistence.path``: serviette's artifacts at the top level, the
Pathway engine confined to the ``PStorage`` subdirectory (the engine logs an
ERROR for every foreign entry in the directory it is given), and in-place
migration of the pre-``PStorage`` layout."""

from __future__ import annotations

import json

from serviette.config.schema import load_config_dict
from serviette.indexer.fingerprint import FINGERPRINT_FILENAME, check_fingerprint
from serviette.indexer.graph import prepare_persistence_dir


def _config(tmp_path):
    return load_config_dict(
        {
            "sources": [{"type": "fs", "path": "/data"}],
            "vector_db": {"type": "duckdb", "path": str(tmp_path / "x.duckdb")},
            "embedder": {"type": "mock"},
            "persistence": {"enabled": True, "path": str(tmp_path / "persist")},
        }
    )


def test_engine_dir_is_a_subdirectory_of_the_data_dir(tmp_path):
    config = _config(tmp_path)
    engine_dir = prepare_persistence_dir(config)
    assert engine_dir == tmp_path / "persist" / "PStorage"
    assert engine_dir.is_dir()
    assert config.persistence.engine_path() == engine_dir


def test_fingerprint_stays_out_of_the_engine_dir(tmp_path):
    config = _config(tmp_path)
    check_fingerprint(config)
    engine_dir = prepare_persistence_dir(config)
    assert (tmp_path / "persist" / FINGERPRINT_FILENAME).is_file()
    assert list(engine_dir.iterdir()) == []


def test_legacy_layout_is_migrated_in_place(tmp_path, caplog):
    """A directory written by serviette <= 0.1.2 (engine state at the top
    level) moves into ``PStorage`` so the incremental state survives the
    upgrade; the fingerprint stays where it is."""

    data_dir = tmp_path / "persist"
    (data_dir / "streams" / "1").mkdir(parents=True)
    (data_dir / "streams" / "1" / "chunk").write_bytes(b"state")
    (data_dir / "runtime_calls").mkdir()
    (data_dir / "1-0-0").write_bytes(b"snapshot")
    (data_dir / FINGERPRINT_FILENAME).write_text(json.dumps({"splitter": {}}))

    engine_dir = prepare_persistence_dir(_config(tmp_path))

    assert (engine_dir / "streams" / "1" / "chunk").read_bytes() == b"state"
    assert (engine_dir / "runtime_calls").is_dir()
    assert (engine_dir / "1-0-0").read_bytes() == b"snapshot"
    assert (data_dir / FINGERPRINT_FILENAME).is_file()
    assert not (engine_dir / FINGERPRINT_FILENAME).exists()
    assert {p.name for p in data_dir.iterdir()} == {FINGERPRINT_FILENAME, "PStorage"}
    assert "Moved the Pathway persistence state" in caplog.text


def test_migration_runs_once(tmp_path):
    data_dir = tmp_path / "persist"
    (data_dir / "streams").mkdir(parents=True)
    config = _config(tmp_path)
    prepare_persistence_dir(config)
    # A later top-level stray (e.g. a user's note) must not be swept into
    # PStorage once the layout exists.
    (data_dir / "notes.txt").write_text("mine")
    prepare_persistence_dir(config)
    assert (data_dir / "notes.txt").is_file()
    assert not (data_dir / "PStorage" / "notes.txt").exists()
