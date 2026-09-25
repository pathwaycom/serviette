"""Byte fetches are retried with backoff; exhausted retries and parser
failures surface as ParseError (the UDF turns that into empty text + ERROR)."""

from __future__ import annotations

import pytest

from serviette.indexer import graph
from serviette.indexer.graph import ParseError, ParserRegistry, fetch_with_retries


class _FlakyFetcher:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def fetch(self, metadata):
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError(f"boom #{self.calls}")
        return b"payload", ".txt"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    delays: list[float] = []
    monkeypatch.setattr(graph.time, "sleep", delays.append)
    return delays


def test_fetch_retries_then_succeeds(_no_sleep):
    fetcher = _FlakyFetcher(failures=2)
    assert fetch_with_retries(fetcher, {"path": "/x.txt"}, retries=3) == (b"payload", ".txt")
    assert fetcher.calls == 3
    assert _no_sleep == [1.0, 2.0]  # exponential backoff between attempts


def test_fetch_gives_up_after_retries_with_parse_error(_no_sleep):
    fetcher = _FlakyFetcher(failures=10)
    with pytest.raises(ParseError, match=r"could not fetch /x\.txt after 3 attempt"):
        fetch_with_retries(fetcher, {"path": "/x.txt"}, retries=2)
    assert fetcher.calls == 3


def test_zero_retries_is_a_single_attempt(_no_sleep):
    fetcher = _FlakyFetcher(failures=1)
    with pytest.raises(ParseError):
        fetch_with_retries(fetcher, {"path": "/x.txt"}, retries=0)
    assert fetcher.calls == 1 and _no_sleep == []


def test_parser_failure_raises_instead_of_empty_text(monkeypatch):
    """A parser exception escapes ``parse`` as ParseError so the caller can
    log which object failed, instead of a silent ``""``."""

    class _BrokenParser:
        def __wrapped__(self, contents):
            raise ValueError("corrupt")

    reg = ParserRegistry()
    monkeypatch.setattr(reg, "_get", lambda kind, options: _BrokenParser())
    with pytest.raises(ParseError, match=r"'utf8' parser failed on 'a.txt': corrupt"):
        reg.parse(b"x", ".txt", "a.txt")


def test_skip_is_still_empty_text_not_an_error(monkeypatch):
    monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
    assert ParserRegistry().parse(b"\x00", ".mp4", "demo.mp4") == ""
