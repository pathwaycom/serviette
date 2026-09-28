"""``splitter`` extra keys must reach the xpack splitter (or be rejected), never
be silently dropped: the schema allows them, and the fingerprint stores them,
so a key with no effect would still trigger a re-index prompt when changed.

Builds the splitter objects only — no graph is run.
"""

from __future__ import annotations

import pytest

from serviette.config.schema import SplitterConfig


def _build(cfg: SplitterConfig):
    pytest.importorskip("pathway")
    from serviette.indexer.graph import build_xpack_splitter

    return build_xpack_splitter(cfg)


def test_token_count_forwards_extra_keys():
    splitter = _build(
        SplitterConfig(type="token_count", chunk_size=256, min_tokens=100, encoding_name="cl100k_base")
    )
    assert splitter.kwargs["max_tokens"] == 256
    assert splitter.kwargs["min_tokens"] == 100
    assert splitter.kwargs["encoding_name"] == "cl100k_base"


def test_recursive_forwards_extra_keys():
    splitter = _build(
        SplitterConfig(
            type="recursive",
            chunk_size=300,
            chunk_overlap=30,
            separators=["\n\n", "\n"],
            is_separator_regex=False,
        )
    )
    assert splitter.kwargs["chunk_size"] == 300
    assert splitter.kwargs["chunk_overlap"] == 30
    assert splitter.kwargs["separators"] == ["\n\n", "\n"]


def test_unknown_splitter_key_fails_at_build_time():
    with pytest.raises(TypeError, match="not_a_real_option"):
        _build(SplitterConfig(type="token_count", not_a_real_option=1))


def test_yaml_null_selects_the_null_splitter(tmp_path):
    """``type: null`` is what people write in YAML; the parser hands it over
    as None, which used to fail validation ("expected a string")."""

    import yaml

    from serviette.config.schema import SplitterConfig, load_config

    assert SplitterConfig(type=None).type == "null"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sources": [{"type": "fs", "path": str(tmp_path)}],
                "vector_db": {"type": "duckdb", "path": str(tmp_path / "e.duckdb")},
                "embedder": {"type": "mock"},
                "splitter": {"type": None},
            }
        )
    )
    assert "type: null" in config_path.read_text()
    config = load_config(config_path)
    assert config.splitter.type == "null"
    # Same fingerprint whether written as null or "null": no spurious
    # "configuration changed" prompt from the spelling alone.
    assert config.splitter.model_dump() == SplitterConfig(type="null").model_dump()


def test_null_splitter_keeps_the_whole_document():
    from pathway.xpacks.llm import splitters

    from serviette.config.schema import SplitterConfig
    from serviette.indexer.graph import build_xpack_splitter

    splitter = build_xpack_splitter(SplitterConfig(type=None))
    assert isinstance(splitter, splitters.NullSplitter)
    text = "para one\n\n" + "word " * 2000
    assert [chunk for chunk, _ in splitter.chunk(text)] == [text]
