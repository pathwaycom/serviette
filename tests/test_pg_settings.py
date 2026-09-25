"""Unit tests for the postgres URL -> libpq settings translation."""

from __future__ import annotations

from serviette.indexer.graph import _libpq_settings


def test_basic_url():
    assert _libpq_settings("postgresql://u:p%40ss@db.example:5433/rag") == {
        "host": "db.example",
        "port": 5433,
        "user": "u",
        "password": "p@ss",
        "dbname": "rag",
    }


def test_query_parameters_pass_through():
    settings = _libpq_settings(
        "postgresql://u:p@localhost/rag?sslmode=require&connect_timeout=5"
    )
    assert settings["sslmode"] == "require"
    assert settings["connect_timeout"] == "5"
    assert settings["dbname"] == "rag"


def test_url_parts_win_over_duplicated_query_keys():
    settings = _libpq_settings("postgresql://u:p@h/db?host=other&dbname=x")
    assert settings["host"] == "h" and settings["dbname"] == "db"
