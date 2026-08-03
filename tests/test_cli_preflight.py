"""The engine-facing commands must fail helpfully on a too-old pathway."""

from __future__ import annotations

import sys
import types

import pytest


def test_old_pathway_gets_instructions(monkeypatch):
    from serviette import cli

    fake = types.ModuleType("pathway")
    fake.__version__ = "0.31.1"
    fake.io = types.SimpleNamespace()  # pre-0.32.1 build: no io.duckdb
    monkeypatch.setitem(sys.modules, "pathway", fake)

    with pytest.raises(SystemExit) as exc:
        cli._ensure_pathway_connectors()
    message = str(exc.value)
    assert "0.31.1" in message
    assert "pathway>=0.32.1" in message
    assert "pip install" in message


def test_current_pathway_passes(monkeypatch):
    from serviette import cli

    fake = types.ModuleType("pathway")
    fake.__version__ = "0.32.1"
    fake.io = types.SimpleNamespace(duckdb=object())
    monkeypatch.setitem(sys.modules, "pathway", fake)

    cli._ensure_pathway_connectors()  # no exception
