"""``${VAR}`` interpolation in configs: an unset variable is a startup error
naming the variable and where it is used — not an empty string that later
surfaces as a provider 401 ("incorrect API key") with no hint of the cause."""

from __future__ import annotations

import pytest
import yaml

from serviette.config.schema import (
    MissingEnvVarError,
    interpolate_env,
    load_config,
    load_config_dict,
)

CONFIG = {
    "sources": [{"type": "fs", "path": "/data"}],
    "vector_db": {"type": "duckdb", "path": "e.duckdb"},
    "embedder": {"type": "openai", "api_key": "${SERVIETTE_TEST_KEY}"},
}


def test_set_variables_are_substituted(monkeypatch):
    monkeypatch.setenv("SERVIETTE_TEST_KEY", "sk-abc")
    assert interpolate_env({"a": "${SERVIETTE_TEST_KEY}", "b": ["x-${SERVIETTE_TEST_KEY}"]}) == {
        "a": "sk-abc",
        "b": ["x-sk-abc"],
    }


def test_unset_variable_is_an_error_naming_it_and_its_place(monkeypatch):
    monkeypatch.delenv("SERVIETTE_TEST_KEY", raising=False)
    monkeypatch.delenv("SERVIETTE_OTHER", raising=False)
    data = {
        "embedder": {"api_key": "${SERVIETTE_TEST_KEY}"},
        "sources": [{"secret": "${SERVIETTE_OTHER}"}],
    }
    with pytest.raises(MissingEnvVarError) as info:
        interpolate_env(data)
    message = str(info.value)
    # Every missing variable is listed once, with the config key using it.
    assert "${SERVIETTE_TEST_KEY} (used in embedder.api_key)" in message
    assert "${SERVIETTE_OTHER} (used in sources[0].secret)" in message
    assert "export" in message


def test_empty_but_set_variable_is_not_missing(monkeypatch):
    monkeypatch.setenv("SERVIETTE_TEST_KEY", "")
    assert interpolate_env("${SERVIETTE_TEST_KEY}") == ""


def test_non_strict_keeps_the_old_empty_string_behavior(monkeypatch):
    monkeypatch.delenv("SERVIETTE_TEST_KEY", raising=False)
    assert interpolate_env("k=${SERVIETTE_TEST_KEY}", strict=False) == "k="
    assert load_config_dict(CONFIG, strict_env=False).embedder.api_key == ""


def test_load_config_reports_the_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SERVIETTE_TEST_KEY", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    with pytest.raises(MissingEnvVarError, match=r"config\.yaml") as info:
        load_config(path)
    assert "${SERVIETTE_TEST_KEY}" in str(info.value)
    # The dedicated error is still a ValueError for callers catching that.
    assert isinstance(info.value, ValueError)


def test_load_config_succeeds_once_the_variable_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("SERVIETTE_TEST_KEY", "sk-abc")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    assert load_config(path).embedder.api_key == "sk-abc"
