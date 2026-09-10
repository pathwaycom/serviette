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
