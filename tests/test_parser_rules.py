"""Unit tests for ``parser:`` rule routing (no Pathway runtime needed)."""

from __future__ import annotations

import pytest

from serviette.config.schema import ParserRule
from serviette.indexer.graph import ParserRegistry


def _registry(*rules: dict) -> ParserRegistry:
    return ParserRegistry([ParserRule(**r) for r in rules])


def test_extension_rule_matches_by_name():
    reg = _registry({"match": ["*.pdf"], "type": "pypdf"})
    assert reg._route(".pdf", "report.pdf")[0] == "pypdf"
    assert reg._route(".pdf", "report.pdf", "/data/docs/report.pdf")[0] == "pypdf"


def test_folder_rule_matches_by_path():
    """fs/s3/sharepoint/pyfilesystem sources report a path, not a name; a
    rule on the folder must select the parser for files inside it."""

    reg = _registry({"match": ["*/scans/*.png"], "type": "utf8"})
    kind, _ = reg._route(".png", "page1.png", "/data/docs/scans/page1.png")
    assert kind == "utf8"
    kind, _ = reg._route(".png", "logo.png", "/data/docs/branding/logo.png")
    assert kind != "utf8"  # falls through to the image default


def test_name_prefix_rule_matches_basename_of_path():
    reg = _registry({"match": ["invoice_*.pdf"], "type": "skip"})
    assert reg._route(".pdf", "invoice_42.pdf", "/data/in/invoice_42.pdf")[0] == "skip"
    assert reg._route(".pdf", "report.pdf", "/data/in/report.pdf")[0] != "skip"


def test_no_name_no_path_still_routes_by_extension():
    reg = _registry({"match": ["*.pdf"], "type": "skip"})
    assert reg._route(".pdf", "")[0] == "skip"


def test_dict_valued_options_are_accepted(monkeypatch):
    """xpack parsers take nested options (docling's ``pdf_pipeline_options``,
    unstructured's ``partition_kwargs``); the registry's instance cache must
    not choke on the unhashable values."""

    built: list[dict] = []

    class FakeParser:
        def __init__(self, **options):
            built.append(options)

    from pathway.xpacks.llm import parsers

    monkeypatch.setattr(parsers, "PypdfParser", FakeParser)
    options = {"pdf_pipeline_options": {"do_ocr": False, "langs": ["en"]}}
    reg = _registry({"match": ["*.pdf"], "type": "pypdf", "options": options})
    reg.check_rule_deps()
    assert reg._get("pypdf", dict(options)) is reg._get("pypdf", dict(options))
    assert built == [options]  # built once at startup, reused afterwards


def test_bad_option_fails_at_startup_not_in_the_pipeline(monkeypatch):
    class FakeParser:
        def __init__(self, *, good: int = 0):
            pass

    from pathway.xpacks.llm import parsers

    monkeypatch.setattr(parsers, "PypdfParser", FakeParser)
    reg = _registry({"match": ["*.pdf"], "type": "pypdf", "options": {"bogus": 1}})
    with pytest.raises(ValueError, match="bogus"):
        reg.check_rule_deps()


def test_skip_rules_are_not_instantiated():
    _registry({"match": ["*.tmp"], "type": "skip", "options": {"reason": "x"}}).check_rule_deps()


def test_default_parser_constructor_failure_skips_the_format(monkeypatch, caplog):
    """No explicit rule guards a default parser, so a constructor failing in
    this environment (missing model, unreachable API) must skip the format
    with a warning — never propagate out of the parse UDF."""

    class Broken:
        def __init__(self, **_):
            raise RuntimeError("model download failed")

    from pathway.xpacks.llm import parsers

    monkeypatch.setattr(parsers, "PypdfParser", Broken)
    reg = ParserRegistry()
    reg._defaults["pdf"] = ("pypdf", {})
    assert reg.parse(b"%PDF", ".pdf", "a.pdf") == ""
    assert reg.parse(b"%PDF", ".pdf", "b.pdf") == ""
    assert caplog.text.count("model download failed") == 1
