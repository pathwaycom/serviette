"""Unit tests for the chunk-id key (no Pathway runtime needed)."""

from __future__ import annotations

import json

import pathway as pw

from serviette.indexer.graph import _metadata_as_json, _metadata_key_json

_META = {
    "created_at": 1,
    "modified_at": 2,
    "owner": "sergey",
    "path": "/docs/a.txt",
    "seen_at": 3,
    "size": 4,
}


def test_key_json_drops_only_the_observation_time():
    key = json.loads(_metadata_key_json.__wrapped__(pw.Json(_META)))
    assert key == {k: v for k, v in _META.items() if k != "seen_at"}


def test_stored_json_keeps_seen_at():
    stored = json.loads(_metadata_as_json.__wrapped__(pw.Json(_META)))
    assert stored == _META


def test_key_json_is_stable_across_observations():
    later = {**_META, "seen_at": 99}
    assert _metadata_key_json.__wrapped__(pw.Json(_META)) == _metadata_key_json.__wrapped__(
        pw.Json(later)
    )
    assert _metadata_key_json.__wrapped__(pw.Json(_META)) != _metadata_key_json.__wrapped__(
        pw.Json({**_META, "modified_at": 5})
    )
