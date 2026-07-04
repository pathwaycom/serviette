"""Source connectors and byte fetchers for the indexer.

We only support sources that have an ``only_metadata`` read mode (like the
filesystem connector): the Pathway graph holds just paths/identifiers and
metadata, never the file bytes, and the bytes are fetched on demand inside the
parse UDF. Today that is:

- **fs** — ``pw.io.fs.read(format="only_metadata")``; bytes read from the local
  path in the metadata.
- **gdrive** — ``pw.io.gdrive.read(format="only_metadata")``; bytes downloaded by
  file ``id`` via the Google Drive API.
- **s3** — ``pw.io.s3.read(format="only_metadata")`` (Pathway develop,
  #10483); bytes downloaded by object key via boto3. Covers any S3-compatible
  store (MinIO, DigitalOcean, Wasabi) through ``endpoint``/``with_path_style``.
- **sharepoint** — ``pathway.xpacks.connectors.sharepoint.read(
  format="only_metadata")`` (same Pathway change; Scale license); bytes
  downloaded by server-relative path via the same certificate-authenticated
  ``office365`` client.

- **pyfilesystem** — ``pw.io.pyfilesystem.read(format="only_metadata")``;
  anything the PyFilesystem library opens (FTP, SFTP, WebDAV, ZIP/TAR,
  ``osfs://``, ``mem://``); bytes re-read from the same filesystem by path.

Each source type provides a :class:`Fetcher` that turns a metadata dict into
``(bytes, suffix)`` for the parser. The fetcher is what makes the parse step
source-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import pathway as pw

# Drive scope sufficient to download file contents.
_GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# ---------------------------------------------------------------------------
# Reading source tables in only_metadata mode
# ---------------------------------------------------------------------------


def read_source(src, name: str) -> pw.Table:
    """Return an ``only_metadata`` table for ``src`` (one ``_metadata`` column).

    ``name`` is the persistable connector name (unique per source).
    """

    if src.type == "fs":
        return pw.io.fs.read(
            src.path,
            format="only_metadata",
            mode=src.mode,
            object_pattern=src.glob,
            max_backlog_size=src.max_backlog_size,
            name=name,
        )
    if src.type == "gdrive":
        return pw.io.gdrive.read(
            src.object_id,
            format="only_metadata",
            mode=src.mode,
            service_user_credentials_file=src.service_user_credentials_file,
            file_name_pattern=src.file_name_pattern,
            max_backlog_size=src.max_backlog_size,
            name=name,
        )
    if src.type == "s3":
        return pw.io.s3.read(
            src.path,
            format="only_metadata",
            aws_s3_settings=pw.io.s3.AwsS3Settings(
                bucket_name=src.bucket,
                access_key=src.access_key,
                secret_access_key=src.secret_access_key,
                region=src.region,
                endpoint=src.endpoint,
                with_path_style=src.with_path_style,
            ),
            mode=src.mode,
            max_backlog_size=src.max_backlog_size,
            name=name,
        )
    if src.type == "pyfilesystem":
        from fs import open_fs

        return pw.io.pyfilesystem.read(
            open_fs(src.fs_url),
            path=src.path,
            format="only_metadata",
            mode=src.mode,
            refresh_interval=float(src.refresh_interval),
            max_backlog_size=src.max_backlog_size,
            name=name,
        )
    if src.type == "sharepoint":
        # NOTE: the SharePoint xpack connector has no max_backlog_size (yet).
        # Requires a Pathway Scale license; the connector polls the site every
        # refresh_interval seconds and emits native additions/retractions.
        from pathway.xpacks.connectors import sharepoint

        return sharepoint.read(
            src.url,
            tenant=src.tenant,
            client_id=src.client_id,
            cert_path=src.cert_path,
            thumbprint=src.thumbprint,
            root_path=src.root_path,
            mode=src.mode,
            format="only_metadata",
            recursive=src.recursive,
            object_size_limit=src.object_size_limit,
            refresh_interval=src.refresh_interval,
        )
    raise ValueError(f"Unsupported source type: {src.type!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Byte fetchers
# ---------------------------------------------------------------------------


class Fetcher(Protocol):
    def fetch(self, metadata: dict[str, Any]) -> tuple[bytes, str]:
        """Return ``(contents, suffix)`` for the object described by ``metadata``."""


class FsFetcher:
    """Read bytes from the local filesystem path in the metadata."""

    def fetch(self, metadata: dict[str, Any]) -> tuple[bytes, str]:
        path = metadata["path"]
        return Path(path).read_bytes(), Path(path).suffix


class GDriveFetcher:
    """Download bytes from Google Drive by file id.

    ``download_fn`` is injectable for tests; in production it lazily builds a
    Drive client from the service-account credentials file.
    """

    def __init__(
        self,
        service_user_credentials_file: str,
        download_fn: Callable[[str], bytes] | None = None,
    ) -> None:
        self._credentials_file = service_user_credentials_file
        self._download_fn = download_fn
        self._service = None

    def _service_download(self, file_id: str) -> bytes:
        if self._service is None:
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build

            creds = Credentials.from_service_account_file(
                self._credentials_file, scopes=_GDRIVE_SCOPES
            )
            self._service = build("drive", "v3", credentials=creds)
        # get_media's execute() returns the raw file bytes.
        return self._service.files().get_media(fileId=file_id).execute()

    def fetch(self, metadata: dict[str, Any]) -> tuple[bytes, str]:
        file_id = metadata["id"]
        download = self._download_fn or self._service_download
        contents = download(file_id)
        # Derive the extension from the file name; Google-native docs (Docs/
        # Sheets) have no binary download and are skipped upstream.
        suffix = Path(metadata.get("name", "")).suffix
        return contents, suffix


class S3Fetcher:
    """Download object bytes from S3 (or an S3-compatible endpoint) by key.

    The ``only_metadata`` row's ``path`` field is the object key. ``client``
    is injectable for tests; in production a ``boto3`` client is built lazily
    from the source settings (boto3 ships with pathway).
    """

    def __init__(self, src, client=None) -> None:
        self._src = src
        self._client = client

    def _ensure_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            src = self._src
            kwargs: dict[str, Any] = {}
            if src.access_key:
                kwargs["aws_access_key_id"] = src.access_key
            if src.secret_access_key:
                kwargs["aws_secret_access_key"] = src.secret_access_key
            if src.region:
                kwargs["region_name"] = src.region
            if src.endpoint:
                kwargs["endpoint_url"] = src.endpoint
            if src.with_path_style:
                kwargs["config"] = Config(s3={"addressing_style": "path"})
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def fetch(self, metadata: dict[str, Any]) -> tuple[bytes, str]:
        key = metadata["path"]
        response = self._ensure_client().get_object(Bucket=self._src.bucket, Key=key)
        return response["Body"].read(), Path(key).suffix


class SharePointFetcher:
    """Download file bytes from SharePoint by server-relative path.

    Reuses the same certificate authentication as the connector
    (``office365`` ``ClientContext``); the ``only_metadata`` row's ``path``
    field is the file's server-relative URL. ``download_fn`` is injectable
    for tests.
    """

    def __init__(self, src, download_fn: Callable[[str], bytes] | None = None) -> None:
        self._src = src
        self._download_fn = download_fn
        self._context = None

    def _ensure_context(self):
        if self._context is None:
            from office365.sharepoint.client_context import ClientContext

            src = self._src
            self._context = ClientContext(src.url).with_client_certificate(
                tenant=src.tenant,
                client_id=src.client_id,
                thumbprint=src.thumbprint,
                cert_path=src.cert_path,
            )
        return self._context

    def _context_download(self, path: str) -> bytes:
        context = self._ensure_context()
        file = context.web.get_file_by_server_relative_path(path)
        return file.get_content().execute_query_retry().value

    def fetch(self, metadata: dict[str, Any]) -> tuple[bytes, str]:
        path = metadata["path"]
        download = self._download_fn or self._context_download
        return download(path), Path(path).suffix


class PyFilesystemFetcher:
    """Read bytes back from the PyFilesystem source by in-filesystem path.

    A separate ``open_fs`` handle is created lazily per process (parse UDFs
    run in every worker); PyFilesystem FS objects are thread-safe.
    ``fs_factory`` is injectable for tests.
    """

    def __init__(self, src, fs_factory: Callable[[], Any] | None = None) -> None:
        self._src = src
        self._fs_factory = fs_factory
        self._fs = None

    def _ensure_fs(self):
        if self._fs is None:
            if self._fs_factory is not None:
                self._fs = self._fs_factory()
            else:
                from fs import open_fs

                self._fs = open_fs(self._src.fs_url)
        return self._fs

    def fetch(self, metadata: dict[str, Any]) -> tuple[bytes, str]:
        path = metadata["path"]
        return self._ensure_fs().readbytes(path), Path(path).suffix


def make_fetcher(src) -> Fetcher:
    if src.type == "fs":
        return FsFetcher()
    if src.type == "gdrive":
        return GDriveFetcher(src.service_user_credentials_file)
    if src.type == "s3":
        return S3Fetcher(src)
    if src.type == "sharepoint":
        return SharePointFetcher(src)
    if src.type == "pyfilesystem":
        return PyFilesystemFetcher(src)
    raise ValueError(f"Unsupported source type: {src.type!r}")  # pragma: no cover
