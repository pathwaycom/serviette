"""The engine-facing commands must fail helpfully on a released pathway."""

from __future__ import annotations

import sys
import types

import pytest


def test_released_pathway_gets_instructions(monkeypatch):
    from vetosh import cli

    fake = types.ModuleType("pathway")
    fake.__version__ = "0.31.1"
    fake.io = types.SimpleNamespace()  # released build: no io.duckdb
    monkeypatch.setitem(sys.modules, "pathway", fake)

    with pytest.raises(SystemExit) as exc:
        cli._ensure_dev_pathway()
    message = str(exc.value)
    assert "0.31.1" in message
    assert "--prerelease=allow" in message
    assert "packages.pathway.com" in message


def test_dev_pathway_passes(monkeypatch):
    from vetosh import cli

    fake = types.ModuleType("pathway")
    fake.__version__ = "0.31.2-dev898"
    fake.io = types.SimpleNamespace(duckdb=object())
    monkeypatch.setitem(sys.modules, "pathway", fake)

    cli._ensure_dev_pathway()  # no exception
