"""Unit tests for ``parser:`` rule routing (no Pathway runtime needed)."""

from __future__ import annotations

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
