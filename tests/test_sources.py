"""Tests for multi-source support (filesystem + Google Drive, only_metadata)."""

from __future__ import annotations

import pytest

from vetosh.config.schema import FsSource, GDriveSource, S3Source, SharePointSource, VetoshConfig
from vetosh.indexer.sources import (
    FsFetcher,
    GDriveFetcher,
    S3Fetcher,
    SharePointFetcher,
    make_fetcher,
)


# -- schema ------------------------------------------------------------------


def test_gdrive_source_validates():
    src = GDriveSource(
        object_id="abc123",
        service_user_credentials_file="./creds.json",
        file_name_pattern="*.pdf",
    )
    assert src.type == "gdrive"
    assert src.mode == "streaming"


def test_mixed_sources_discriminated():
    config = VetoshConfig.model_validate(
        {
            "sources": [
                {"type": "fs", "path": "/data/docs", "glob": "**/*"},
                {
                    "type": "gdrive",
                    "object_id": "folder-id",
                    "service_user_credentials_file": "./creds.json",
                },
            ],
            "vector_db": {"type": "duckdb", "path": "/tmp/x.duckdb"},
            "embedder": {"type": "mock"},
        }
    )
    config.for_indexer()
    assert isinstance(config.sources[0], FsSource)
    assert isinstance(config.sources[1], GDriveSource)


# -- fetchers ----------------------------------------------------------------


def test_fs_fetcher_reads_bytes_and_suffix(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes(b"hello world")
    contents, suffix = FsFetcher().fetch({"path": str(path)})
    assert contents == b"hello world"
    assert suffix == ".txt"


def test_gdrive_fetcher_uses_injected_download():
    calls = []

    def fake_download(file_id: str) -> bytes:
        calls.append(file_id)
        return b"%PDF-fake-bytes"

    fetcher = GDriveFetcher("./creds.json", download_fn=fake_download)
    contents, suffix = fetcher.fetch({"id": "file-42", "name": "report.pdf"})
    assert contents == b"%PDF-fake-bytes"
    assert suffix == ".pdf"
    assert calls == ["file-42"]


def test_s3_source_validates():
    src = S3Source(
        bucket="docs",
        path="reports/",
        endpoint="http://localhost:9000",
        region="us-east-1",
        with_path_style=True,
    )
    assert src.type == "s3" and src.mode == "streaming"


def test_s3_custom_endpoint_requires_region():
    import pytest

    with pytest.raises(ValueError, match="region"):
        S3Source(bucket="docs", endpoint="http://localhost:9000")


def test_sharepoint_source_validates():
    src = SharePointSource(
        url="https://co.sharepoint.com/sites/X",
        tenant="tenant-guid",
        client_id="client-guid",
        cert_path="./cert.pem",
        thumbprint="ABC",
        root_path="Shared Documents",
    )
    assert src.type == "sharepoint" and src.recursive


class _FakeS3Client:
    def __init__(self):
        self.calls = []

    def get_object(self, Bucket, Key):  # noqa: N803 - boto3 API
        import io

        self.calls.append((Bucket, Key))
        return {"Body": io.BytesIO(b"s3 file bytes")}


def test_s3_fetcher_downloads_by_key():
    client = _FakeS3Client()
    fetcher = S3Fetcher(S3Source(bucket="docs"), client=client)
    contents, suffix = fetcher.fetch({"path": "reports/q1.pdf"})
    assert contents == b"s3 file bytes"
    assert suffix == ".pdf"
    assert client.calls == [("docs", "reports/q1.pdf")]


def test_sharepoint_fetcher_uses_injected_download():
    calls = []

    def fake_download(path: str) -> bytes:
        calls.append(path)
        return b"docx-bytes"

    src = SharePointSource(
        url="https://co.sharepoint.com/sites/X",
        tenant="t",
        client_id="c",
        cert_path="./cert.pem",
        thumbprint="ABC",
        root_path="Shared Documents",
    )
    contents, suffix = SharePointFetcher(src, download_fn=fake_download).fetch(
        {"path": "/sites/X/Shared Documents/plan.docx"}
    )
    assert contents == b"docx-bytes"
    assert suffix == ".docx"
    assert calls == ["/sites/X/Shared Documents/plan.docx"]


def test_make_fetcher_dispatch():
    assert isinstance(make_fetcher(FsSource(path="/x")), FsFetcher)
    gd = make_fetcher(
        GDriveSource(object_id="x", service_user_credentials_file="./c.json")
    )
    assert isinstance(gd, GDriveFetcher)
    assert isinstance(make_fetcher(S3Source(bucket="b")), S3Fetcher)
    sp = make_fetcher(
        SharePointSource(
            url="https://co.sharepoint.com/sites/X",
            tenant="t",
            client_id="c",
            cert_path="./c.pem",
            thumbprint="A",
            root_path="Docs",
        )
    )
    assert isinstance(sp, SharePointFetcher)


def test_monitoring_http_port_config():
    """indexer.monitoring_http_port: validated, defaults to disabled."""
    import pytest
    from pydantic import ValidationError

    from vetosh.config.schema import IndexerConfig

    assert IndexerConfig().monitoring_http_port is None
    assert IndexerConfig(monitoring_http_port=20500).monitoring_http_port == 20500
    with pytest.raises(ValidationError):
        IndexerConfig(monitoring_http_port=0)
    with pytest.raises(ValidationError):
        IndexerConfig(monitoring_http_port=70000)


def test_pyfilesystem_source_validates():
    from vetosh.config.schema import PyFilesystemSource

    src = PyFilesystemSource(fs_url="ftp://user:pw@host/dir")
    assert src.path == "" and src.mode == "streaming"
    assert src.refresh_interval == 30 and src.max_backlog_size == 1000
    config = VetoshConfig.model_validate(
        {
            "sources": [{"type": "pyfilesystem", "fs_url": "zip://docs.zip", "path": "a"}],
            "vector_db": {"type": "duckdb", "path": "x.duckdb"},
            "embedder": {"type": "mock"},
        }
    )
    assert config.sources[0].fs_url == "zip://docs.zip"


def test_pyfilesystem_fetcher_reads_bytes_and_suffix():
    # The pyfilesystem source is an optional extra; the test mirrors that.
    pyfs = pytest.importorskip("fs")

    from vetosh.config.schema import PyFilesystemSource
    from vetosh.indexer.sources import PyFilesystemFetcher, make_fetcher

    mem = pyfs.open_fs("mem://")
    mem.makedirs("docs")
    mem.writebytes("docs/report.pdf", b"%PDF fake")

    src = PyFilesystemSource(fs_url="mem://")
    fetcher = PyFilesystemFetcher(src, fs_factory=lambda: mem)
    contents, suffix = fetcher.fetch({"path": "docs/report.pdf"})
    assert contents == b"%PDF fake" and suffix == ".pdf"

    assert isinstance(make_fetcher(src), PyFilesystemFetcher)


def _registry(rules=None, env=None, importable=frozenset(), monkeypatch=None):
    """ParserRegistry with a controlled environment for routing tests."""
    import vetosh.indexer.graph as graph_mod

    registry = graph_mod.ParserRegistry(rules)
    registry._importable = staticmethod(lambda module: module in importable)  # type: ignore[method-assign]
    return registry


def test_parser_defaults_prefer_best_keyless(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)

    # Everything installed: docling wins for pdf, paddle for images.
    reg = _registry(importable={"docling", "paddleocr"})
    assert reg._route(".pdf", "a.pdf")[0] == "docling"
    assert reg._route(".png", "scan.png")[0] == "paddle_ocr"
    assert reg._route(".txt", "a.txt")[0] == "utf8"
    assert reg._route(".docx", "a.docx")[0] == "unstructured"
    # Key-requiring modalities are skipped without their keys.
    assert reg._route(".mp3", "a.mp3")[0] == "skip"
    assert reg._route(".mp4", "a.mp4")[0] == "skip"

    # Nothing optional installed: pdf falls back to pypdf, images skip.
    reg = _registry(importable=set())
    assert reg._route(".pdf", "a.pdf")[0] == "pypdf"
    assert reg._route(".png", "scan.png")[0] == "skip"


def test_parser_defaults_enable_keyed_modalities_with_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("TWELVELABS_API_KEY", "k")
    reg = _registry()
    assert reg._route(".wav", "call.wav")[0] == "whisper"
    assert reg._route(".mp4", "demo.mp4")[0] == "twelvelabs_video"


def test_parser_user_rules_win_and_fall_through(monkeypatch):
    from vetosh.config.schema import ParserRule

    monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
    rules = [
        ParserRule(match=["*.pdf"], type="pypdf"),
        ParserRule(
            match=["*.mp4"], type="twelvelabs_video", options={"prompt": "Describe"}
        ),
    ]
    reg = _registry(rules, importable={"docling"})
    # Explicit rule beats the docling default.
    assert reg._route(".pdf", "a.pdf")[0] == "pypdf"
    # Explicit rule enables video even without the env key (key in options/env at runtime).
    kind, options = reg._route(".mp4", "demo.mp4")
    assert kind == "twelvelabs_video" and options["prompt"] == "Describe"
    # Unmatched files fall through to defaults.
    assert reg._route(".txt", "a.txt")[0] == "utf8"


def test_parser_skip_produces_empty_text(monkeypatch, caplog):
    monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
    reg = _registry()
    assert reg.parse(b"\x00fakevideo", ".mp4", "demo.mp4") == ""
    assert any("Skipping" in r.message for r in caplog.records)


def test_parser_rules_in_fingerprint_without_credentials(monkeypatch):
    from vetosh.indexer.fingerprint import build_fingerprint

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
    config = VetoshConfig.model_validate(
        {
            "sources": [{"type": "fs", "path": "./docs"}],
            "vector_db": {"type": "duckdb", "path": "x.duckdb"},
            "embedder": {"type": "mock"},
            "parser": [
                {
                    "match": ["*.mp4"],
                    "type": "twelvelabs_video",
                    "options": {"prompt": "P", "api_key": "SECRET"},
                }
            ],
        }
    )
    fp = build_fingerprint(config)
    dumped = str(fp["parser"])
    assert "twelvelabs_video" in dumped and "P" in dumped
    assert "SECRET" not in dumped


def test_pyfilesystem_source_without_extra_fails_helpfully(monkeypatch):
    """No bare ModuleNotFoundError: the user is told which extra to install."""
    import sys

    import pytest as _pytest

    from vetosh.config.schema import PyFilesystemSource
    from vetosh.indexer.sources import read_source

    monkeypatch.setitem(sys.modules, "fs", None)  # simulates the missing extra
    src = PyFilesystemSource(fs_url="mem://")
    with _pytest.raises(SystemExit) as exc:
        read_source(src, name="s")
    assert "vetosh[pyfilesystem]" in str(exc.value)
