"""Backend-agnostic unit tests for the shared in-process BM25 hybrid mixin
(:class:`serviette.server.hybrid.KeywordHybridMixin`), driven by an in-memory
fake accessor so the fusion / rebuild / size-cap logic is verified without any
real vector store."""

from __future__ import annotations

import asyncio
import logging

from serviette.server.accessors.abstract import AsyncVectorAccessor
from serviette.server.hybrid import KeywordHybridMixin


class _Config:
    def __init__(self, hybrid=False, hybrid_max_chunks=5_000_000, hybrid_refresh_seconds=None):
        self.hybrid = hybrid
        self.hybrid_max_chunks = hybrid_max_chunks
        # None here: the unit tests below pin the timed refresh explicitly.
        self.hybrid_refresh_seconds = hybrid_refresh_seconds


class FakeHybridAccessor(KeywordHybridMixin, AsyncVectorAccessor):
    """A fake store: the "vector" query returns rows in insertion order; the
    hybrid hooks expose the same rows for the BM25 leg."""

    def __init__(self, rows, config):
        # rows: list of (text, embedding) — metadata omitted for brevity.
        self._rows = rows
        self.fetch_all_calls = 0
        self._init_hybrid(config)

    async def retrieve(self, embedding, k):
        return await self.retrieve_ex(embedding, k)

    async def retrieve_ex(self, embedding, k, *, query_text=None, with_embeddings=False):
        # "Vector" order = insertion order, descending fake score.
        vector_hits = []
        for i, (text, emb) in enumerate(self._rows[:k]):
            hit = {"text": text, "metadata": {}, "score": 1.0 - i * 0.01}
            if with_embeddings:
                hit["embedding"] = emb
            vector_hits.append(hit)
        return await self._fuse(vector_hits, query_text, k, with_embeddings)

    async def _hybrid_count(self):
        return len(self._rows)

    async def _hybrid_fetch_all(self, with_embeddings):
        self.fetch_all_calls += 1
        hits = []
        for text, emb in self._rows:
            hit = {"text": text, "metadata": {}}
            if with_embeddings:
                hit["embedding"] = emb
            hits.append(hit)
        return hits

    async def close(self):
        return None


ROWS = [
    ("alpha cats streaming engine", [1.0, 0.0]),
    ("beta dogs framework", [0.0, 1.0]),
    ("gamma Guadalupe Hidalgo 1848 treaty", [0.5, 0.5]),
]


def test_hybrid_off_is_pure_vector():
    acc = FakeHybridAccessor(ROWS, _Config(hybrid=False))
    hits = asyncio.run(acc.retrieve_ex([0.0, 0.0], 2, query_text="Guadalupe Hidalgo"))
    # Pure vector order (insertion), keyword leg never consulted.
    assert [h["text"] for h in hits] == [ROWS[0][0], ROWS[1][0]]
    assert acc.fetch_all_calls == 0


def test_hybrid_surfaces_keyword_match_vector_missed():
    acc = FakeHybridAccessor(ROWS, _Config(hybrid=True))
    # Vector top-2 would be rows 0,1; the keyword leg pulls in the Guadalupe
    # Hidalgo chunk (row 2), which RRF then fuses into the top-2.
    hits = asyncio.run(
        acc.retrieve_ex([0.0, 0.0], 2, query_text="when was Guadalupe Hidalgo signed")
    )
    assert any("Guadalupe Hidalgo" in h["text"] for h in hits)


def test_hybrid_no_query_text_stays_vector():
    acc = FakeHybridAccessor(ROWS, _Config(hybrid=True))
    hits = asyncio.run(acc.retrieve_ex([0.0, 0.0], 2, query_text=None))
    assert [h["text"] for h in hits] == [ROWS[0][0], ROWS[1][0]]
    assert acc.fetch_all_calls == 0


def test_index_rebuilds_only_when_count_changes():
    acc = FakeHybridAccessor(list(ROWS), _Config(hybrid=True))

    async def run():
        await acc.retrieve_ex([0.0, 0.0], 2, query_text="cats")
        await acc.retrieve_ex([0.0, 0.0], 2, query_text="dogs")
        calls_before = acc.fetch_all_calls
        acc._rows.append(("delta new zanzibar spice", [0.2, 0.2]))
        await acc.retrieve_ex([0.0, 0.0], 2, query_text="zanzibar")
        return calls_before

    calls_before = asyncio.run(run())
    # Built once for the first two queries (same count), rebuilt after growth.
    assert calls_before == 1
    assert acc.fetch_all_calls == 2


def test_size_cap_skips_keyword_leg(caplog):
    acc = FakeHybridAccessor(ROWS, _Config(hybrid=True, hybrid_max_chunks=1))
    with caplog.at_level(logging.WARNING):
        hits = asyncio.run(
            acc.retrieve_ex([0.0, 0.0], 2, query_text="Guadalupe Hidalgo")
        )
    # Above the cap: pure vector, one-time warning, no corpus fetch.
    assert [h["text"] for h in hits] == [ROWS[0][0], ROWS[1][0]]
    assert acc.fetch_all_calls == 0
    assert any("hybrid_max_chunks" in r.message for r in caplog.records)


def test_mmr_path_carries_embeddings_through_fusion():
    acc = FakeHybridAccessor(ROWS, _Config(hybrid=True))
    hits = asyncio.run(
        acc.retrieve_ex([0.0, 0.0], 3, query_text="cats", with_embeddings=True)
    )
    # Every fused hit keeps its embedding (needed downstream by MMR).
    assert all("embedding" in h for h in hits)


class _VersionedAccessor(FakeHybridAccessor):
    """A backend that can report a change marker next to the count (as the
    DuckDB / pgvector accessors do with the newest ``seen_at``)."""

    async def _hybrid_version(self):
        return len(self._rows), hash(tuple(text for text, _ in self._rows))


EDITED = ("pricing: Team tier costs 199 EUR per month", [1.0, 0.0])
ORIGINAL = ("pricing: Team tier costs 129 EUR per month", [1.0, 0.0])


def test_change_marker_rebuilds_on_in_place_edit():
    """Same row count, new text: the keyword leg must not keep serving the
    old chunk (the demo's "edit pricing.md" moment)."""

    acc = _VersionedAccessor([ORIGINAL, ROWS[1]], _Config(hybrid=True))

    async def run():
        before = await acc.retrieve_ex([0.0, 0.0], 2, query_text="129 EUR")
        acc._rows[0] = EDITED
        after = await acc.retrieve_ex([0.0, 0.0], 2, query_text="129 EUR")
        return before, after

    before, after = asyncio.run(run())
    assert ORIGINAL[0] in [h["text"] for h in before]
    assert ORIGINAL[0] not in [h["text"] for h in after]
    assert acc.fetch_all_calls == 2


def _clock(monkeypatch, start=1000.0):
    now = {"t": start}
    monkeypatch.setattr("serviette.server.hybrid.time.monotonic", lambda: now["t"])
    return now


def test_timed_refresh_catches_edits_the_count_misses(monkeypatch):
    """Without a change marker the count alone misses an in-place edit; the
    timed refresh bounds how long the stale index is served, and runs in the
    background so the triggering query is answered from the current index."""

    now = _clock(monkeypatch)
    acc = FakeHybridAccessor([ORIGINAL, ROWS[1]], _Config(hybrid=True, hybrid_refresh_seconds=30))

    async def run():
        await acc.retrieve_ex([0.0, 0.0], 2, query_text="129 EUR")
        acc._rows[0] = EDITED
        now["t"] += 10
        stale = await acc.retrieve_ex([0.0, 0.0], 2, query_text="129 EUR")
        assert acc.fetch_all_calls == 1  # within the TTL: no refresh yet
        now["t"] += 25
        during = await acc.retrieve_ex([0.0, 0.0], 2, query_text="129 EUR")
        # Whether the task has already run by now depends on the asyncio
        # scheduling of the Python version (it has on 3.13+ with this instant
        # fake); what matters is that it was scheduled and that ``during``
        # was answered from the old index.
        refresh = acc._bm25_refresh
        assert refresh is not None
        await refresh
        fresh = await acc.retrieve_ex([0.0, 0.0], 2, query_text="129 EUR")
        await acc.close()
        return stale, during, fresh

    stale, during, fresh = asyncio.run(run())
    assert ORIGINAL[0] in [h["text"] for h in stale]
    assert ORIGINAL[0] in [h["text"] for h in during]  # served the old index, rebuilt behind it
    assert ORIGINAL[0] not in [h["text"] for h in fresh]
    assert acc.fetch_all_calls == 2


def test_timed_refresh_disabled_with_none(monkeypatch):
    now = _clock(monkeypatch)
    acc = FakeHybridAccessor(list(ROWS), _Config(hybrid=True, hybrid_refresh_seconds=None))

    async def run():
        await acc.retrieve_ex([0.0, 0.0], 2, query_text="cats")
        now["t"] += 10_000
        await acc.retrieve_ex([0.0, 0.0], 2, query_text="cats")
        assert acc._bm25_refresh is None

    asyncio.run(run())
    assert acc.fetch_all_calls == 1


def test_timed_refresh_runs_one_at_a_time(monkeypatch):
    now = _clock(monkeypatch)
    acc = FakeHybridAccessor(list(ROWS), _Config(hybrid=True, hybrid_refresh_seconds=1))

    async def run():
        await acc.retrieve_ex([0.0, 0.0], 2, query_text="cats")
        now["t"] += 5
        await acc.retrieve_ex([0.0, 0.0], 2, query_text="cats")
        first = acc._bm25_refresh
        await acc.retrieve_ex([0.0, 0.0], 2, query_text="dogs")
        assert acc._bm25_refresh is first  # no second task while one is pending
        await acc.close()

    asyncio.run(run())
    assert acc.fetch_all_calls == 2
