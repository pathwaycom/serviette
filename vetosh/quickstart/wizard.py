"""Interactive terminal wizard: ``vetosh quickstart``.

Generates a vetosh YAML config by asking a short series of questions. In a real
terminal, multiple-choice questions are navigated with the **arrow keys** (↑/↓ to
move, Enter to select) and text questions come with their default **pre-filled
and editable** — press Enter to accept, or edit in place. When stdin is not a TTY
(tests, pipes), it falls back to numbered/typed input with the same semantics.

Document sources are collected in a **loop**: the first source is required, then
you may keep adding more (filesystem today; S3 / SharePoint / Google Drive are
planned and slot into ``SOURCE_TYPES`` + :meth:`Wizard._collect_source` without
touching the rest of the flow).

The config-building logic (:func:`build_config`) is kept pure and separate from
the I/O so it can be unit-tested without a terminal.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import yaml

from vetosh import APP_NAME

# Embedder families supported by pathway.xpacks.llm.embedders.
# The default requires no API key at all (fully local).
EMBEDDER_CHOICES = ["sentence_transformer", "openai", "litellm", "gemini", "bedrock"]
EMBEDDER_LABELS = [
    "sentence_transformer (local \u00b7 no API key needed)",
    "openai",
    "litellm",
    "gemini",
    "bedrock",
]

# Document source types offered by the wizard. Only sources with an
# only_metadata read mode are supported.
SOURCE_TYPES: list[tuple[str, str]] = [
    ("fs", "Filesystem (local directory)"),
    ("gdrive", "Google Drive"),
    ("s3", "S3 / MinIO / S3-compatible bucket"),
    ("sharepoint", "Microsoft SharePoint"),
    ("pyfilesystem", "FTP / SFTP / WebDAV / ZIP (PyFilesystem URL)"),
]

# Default environment-variable references per provider, so generated configs
# never hardcode a credential (the "no hardcoded credentials" principle).
_ENV_REF = {
    "openai": "${OPENAI_API_KEY}",
    "litellm": "${OPENAI_API_KEY}",
    "gemini": "${GEMINI_API_KEY}",
    "bedrock": None,
    "sentence_transformer": None,
}

# Simple, brand-neutral defaults suggested by the wizard.
DEFAULT_COLLECTION = "embeddings"


# ---------------------------------------------------------------------------
# Pure config builder (no I/O)
# ---------------------------------------------------------------------------


def _duckdb_table_exists(path: str, table: str) -> bool:
    try:
        import duckdb

        conn = duckdb.connect(path, read_only=True)
        try:
            tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        finally:
            conn.close()
        return table in tables
    except Exception:  # noqa: BLE001 - unreadable file: let the indexer report
        return False


def build_config(answers: dict[str, Any]) -> dict[str, Any]:
    """Assemble a config dict from collected ``answers``.

    Only the sections relevant to ``answers['config_type']`` are emitted.
    ``answers['sources']`` is a list of fully-formed source dicts (one per source
    the user added in the loop).
    """

    config_type = answers["config_type"]
    needs_indexer = config_type in {"universal", "indexer"}
    needs_server = config_type in {"universal", "server"}

    config: dict[str, Any] = {"pathway_license_key": answers["license_key"]}

    if needs_indexer:
        config["sources"] = answers["sources"]

    config["vector_db"] = _vector_db_section(answers)
    config["embedder"] = _embedder_section(answers)

    if needs_indexer:
        config["splitter"] = {
            "type": "token_count",
            "chunk_size": answers.get("chunk_size", 512),
            "chunk_overlap": answers.get("chunk_overlap", 50),
        }
        # Persistence is a silent default (enabled, on disk, ./persistence):
        # the section is not emitted and the wizard does not ask — advanced
        # users tune it by adding a `persistence:` section to the config.

    if needs_server:
        config["server"] = {
            "host": answers.get("server_host", "0.0.0.0"),
            "port": answers.get("server_port", 8989),
        }
        if answers.get("rag_enabled"):
            config["llm"] = _llm_section(answers)

    return config


def _vector_db_section(answers: dict[str, Any]) -> dict[str, Any]:
    vdb_type = answers["vector_db_type"]
    collection = answers.get("collection", DEFAULT_COLLECTION)
    if vdb_type == "duckdb":
        return {
            "type": "duckdb",
            "path": answers["duckdb_path"],
            "table": collection,
        }
    if vdb_type == "milvus":
        section: dict[str, Any] = {"type": "milvus", "collection": collection}
        if answers.get("milvus_uri"):
            section["uri"] = answers["milvus_uri"]
        else:
            section["host"] = answers.get("milvus_host", "localhost")
            section["port"] = answers.get("milvus_port", 19530)
        return section
    if vdb_type == "qdrant":
        section = {
            "type": "qdrant",
            "host": answers.get("qdrant_host", "localhost"),
            "collection": collection,
        }
        if answers.get("qdrant_api_key"):
            section["api_key"] = answers["qdrant_api_key"]
        return section
    if vdb_type == "chroma":
        return {
            "type": "chroma",
            "host": answers.get("chroma_host", "localhost"),
            "port": answers.get("chroma_port", 8000),
            "collection": collection,
        }
    if vdb_type == "weaviate":
        section = {
            "type": "weaviate",
            "http_host": answers.get("weaviate_host", "localhost"),
            "http_port": answers.get("weaviate_port", 8080),
            "collection": answers.get("weaviate_collection", "VetoshEmbeddings"),
        }
        if answers.get("weaviate_api_key"):
            section["api_key"] = answers["weaviate_api_key"]
        return section
    if vdb_type == "pinecone":
        return {
            "type": "pinecone",
            "index_name": answers["pinecone_index"],
            "api_key": answers.get("pinecone_api_key", "${PINECONE_API_KEY}"),
        }
    if vdb_type == "mongodb":
        return {
            "type": "mongodb",
            "connection_string": answers["mongodb_connection_string"],
            "database": answers["mongodb_database"],
            "collection": collection,
        }
    return {
        "type": "pgvector",
        "connection_string": answers["pg_connection_string"],
        "table": collection,
    }


def _embedder_section(answers: dict[str, Any]) -> dict[str, Any]:
    etype = answers["embedder_type"]
    section: dict[str, Any] = {"type": etype}
    if answers.get("embedder_model"):
        section["model"] = answers["embedder_model"]
    api_key = answers.get("embedder_api_key", _ENV_REF.get(etype))
    if api_key:
        section["api_key"] = api_key
    return section


def _llm_section(answers: dict[str, Any]) -> dict[str, Any]:
    section: dict[str, Any] = {"type": answers.get("llm_type", "openai")}
    if answers.get("llm_model"):
        section["model"] = answers["llm_model"]
    api_key = answers.get("llm_api_key", _ENV_REF.get(answers.get("llm_type", "openai")))
    if api_key:
        section["api_key"] = api_key
    return section


def dump_yaml(config: dict[str, Any]) -> str:
    header = f"# {APP_NAME} config — generated by `{APP_NAME} quickstart`\n"
    return header + yaml.safe_dump(config, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Prompters: arrow-key interactive vs. scripted (tests / non-TTY)
# ---------------------------------------------------------------------------


class Prompter(ABC):
    """Abstracts the three primitives the wizard needs."""

    def info(self, message: str) -> None:  # noqa: D401 - simple passthrough
        print(message)

    @abstractmethod
    def select(self, prompt: str, options: list[str], default_index: int = 0) -> int:
        """Return the index of the chosen option."""

    @abstractmethod
    def text(self, prompt: str, default: str | None = None, *, required: bool = False) -> str:
        """Return a (possibly default/pre-filled) text value."""

    def confirm(self, prompt: str, default: bool = True) -> bool:
        idx = self.select(prompt, ["Yes", "No"], default_index=0 if default else 1)
        return idx == 0


class ScriptedPrompter(Prompter):
    """Numbered/typed prompts driven by injectable I/O (tests, non-TTY)."""

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._input = input_fn
        self._output = output_fn

    def info(self, message: str) -> None:
        self._output(message)

    def select(self, prompt: str, options: list[str], default_index: int = 0) -> int:
        self._output(prompt)
        for i, opt in enumerate(options, start=1):
            marker = " (default)" if i - 1 == default_index else ""
            self._output(f"    [{i}] {opt}{marker}")
        while True:
            raw = self._input(f"  Enter number [1-{len(options)}] [{default_index + 1}]: ").strip()
            if not raw:
                return default_index
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return int(raw) - 1
            self._output("  Invalid choice.")

    def text(self, prompt: str, default: str | None = None, *, required: bool = False) -> str:
        suffix = f" [{default}]" if default else ""
        while True:
            raw = self._input(f"  {prompt}{suffix}: ").strip()
            if raw:
                return raw
            if default is not None and not required:
                return default
            if default:
                return default
            self._output("  This field is required.")


class InteractivePrompter(Prompter):
    """Arrow-key menus and pre-filled editable text for a real terminal."""

    # -- low-level key reading ------------------------------------------------

    @staticmethod
    def _read_key() -> str:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x03":  # Ctrl-C
                raise KeyboardInterrupt
            if ch == "\x04":  # Ctrl-D
                raise EOFError
            if ch in ("\r", "\n"):
                return "enter"
            if ch == "\x1b":  # escape sequence (arrow keys)
                seq = sys.stdin.read(2)
                return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq, "esc")
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def select(self, prompt: str, options: list[str], default_index: int = 0) -> int:
        print(prompt)
        print("  (use ↑/↓ arrows, Enter to select)")
        active = default_index
        # Initial render.
        for i, opt in enumerate(options):
            print(self._render_option(opt, i == active))
        while True:
            key = self._read_key()
            if key == "up":
                active = (active - 1) % len(options)
            elif key == "down":
                active = (active + 1) % len(options)
            elif key.isdigit() and 1 <= int(key) <= len(options):
                active = int(key) - 1
            elif key == "enter":
                # Leave the menu rendered with the final selection.
                return active
            else:
                continue
            # Re-render: move cursor up over the option lines and rewrite.
            sys.stdout.write(f"\x1b[{len(options)}A")
            for i, opt in enumerate(options):
                sys.stdout.write("\x1b[2K" + self._render_option(opt, i == active) + "\n")
            sys.stdout.flush()

    @staticmethod
    def _render_option(label: str, active: bool) -> str:
        if active:
            return f"  \x1b[7m › {label} \x1b[0m"  # reverse video
        return f"    {label}"

    def text(self, prompt: str, default: str | None = None, *, required: bool = False) -> str:
        import readline

        prefill = default or ""

        def hook() -> None:
            readline.insert_text(prefill)
            readline.redisplay()

        while True:
            readline.set_pre_input_hook(hook)
            try:
                value = input(f"  {prompt}: ").strip()
            finally:
                readline.set_pre_input_hook()
            if value:
                return value
            if not required and default is not None:
                return default
            print("  This field is required.")


def _default_prompter() -> Prompter:
    if sys.stdin.isatty() and sys.stdout.isatty():
        return InteractivePrompter()
    return ScriptedPrompter()


# ---------------------------------------------------------------------------
# The wizard flow
# ---------------------------------------------------------------------------


class Wizard:
    def __init__(
        self,
        prompter: Prompter | None = None,
        *,
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[[str], None] | None = None,
    ) -> None:
        if prompter is not None:
            self.p = prompter
        elif input_fn is not None or output_fn is not None:
            self.p = ScriptedPrompter(input_fn or input, output_fn or print)
        else:
            self.p = _default_prompter()
        self._step = 0

    def _section(self, title: str) -> None:
        self._step += 1
        self.p.info(f"\nStep {self._step}: {title}")

    # -- sources loop --------------------------------------------------------

    def _collect_sources(self) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        while True:
            first = not sources
            labels = [label for _, label in SOURCE_TYPES]
            done_index = None
            if not first:
                done_index = len(labels)
                labels = labels + ["Done — no more sources"]
            prompt = "Add a document source:" if first else "Add another source?"
            idx = self.p.select(prompt, labels, default_index=0 if first else done_index)
            if done_index is not None and idx == done_index:
                break
            stype = SOURCE_TYPES[idx][0]
            sources.append(self._collect_source(stype))
        return sources

    def _collect_source(self, stype: str) -> dict[str, Any]:
        if stype == "fs":
            path = self.p.text("Directory path", required=True)
            # glob is an advanced setting: default emitted into the YAML,
            # edit it there.
            return {"type": "fs", "path": path, "glob": "**/*"}
        if stype == "gdrive":
            object_id = self.p.text("Google Drive folder/file id", required=True)
            creds = self.p.text(
                "Service-account credentials JSON file", default="./credentials.json"
            )
            pattern = self.p.text("File name pattern (e.g. *.pdf, blank = all)", default="")
            source: dict[str, Any] = {
                "type": "gdrive",
                "object_id": object_id,
                "service_user_credentials_file": creds,
            }
            if pattern:
                source["file_name_pattern"] = pattern
            return source
        if stype == "s3":
            bucket = self.p.text("Bucket name", required=True)
            prefix = self.p.text("Key prefix to index (blank = whole bucket)", default="")
            endpoint = self.p.text(
                "Custom endpoint (blank for AWS; e.g. http://localhost:9000 for MinIO)",
                default="",
            )
            source = {
                "type": "s3",
                "bucket": bucket,
                "path": prefix,
                "access_key": "${AWS_ACCESS_KEY_ID}",
                "secret_access_key": "${AWS_SECRET_ACCESS_KEY}",
            }
            if endpoint:
                source["endpoint"] = endpoint
                source["with_path_style"] = True  # the norm for self-hosted S3
                # Required with a custom endpoint (signature scope).
                source["region"] = self.p.text("Region", default="us-east-1")
            return source
        if stype == "sharepoint":
            return {
                "type": "sharepoint",
                "url": self.p.text(
                    "Site URL (https://company.sharepoint.com/sites/MySite)", required=True
                ),
                "tenant": self.p.text("Tenant id (GUID)", required=True),
                "client_id": self.p.text("Client id of the app registration", required=True),
                "cert_path": self.p.text("Certificate .pem path", default="./sharepoint.pem"),
                "thumbprint": self.p.text("Certificate thumbprint", required=True),
                "root_path": self.p.text("Root path to index (e.g. Shared Documents)", required=True),
            }
        if stype == "pyfilesystem":
            return {
                "type": "pyfilesystem",
                "fs_url": self.p.text(
                    'Filesystem URL (e.g. "ftp://user:pass@host/dir", "zip://./docs.zip")',
                    required=True,
                ),
                "path": self.p.text("Path inside it to index", default=""),
            }
        raise ValueError(f"Unsupported source type: {stype!r}")  # pragma: no cover

    # -- the full flow -------------------------------------------------------

    def run(self) -> dict[str, Any]:
        answers: dict[str, Any] = {}

        self._section("What to generate?")
        type_idx = self.p.select(
            "Config type:",
            ["Universal config", "Indexer config only", "Server config only"],
            default_index=0,
        )
        config_type = ["universal", "indexer", "server"][type_idx]
        answers["config_type"] = config_type
        needs_indexer = config_type in {"universal", "indexer"}
        needs_server = config_type in {"universal", "server"}

        if needs_indexer:
            self._section("Document sources")
            answers["sources"] = self._collect_sources()

        self._section("Vector database")
        vdb_values = [
            "duckdb",
            "pgvector",
            "qdrant",
            "milvus",
            "chroma",
            "weaviate",
            "pinecone",
            "mongodb",
        ]
        vdb_labels = [
            "duckdb (local file · zero setup · in-database vector search)",
            "pgvector (Postgres)",
            "qdrant",
            "milvus",
            "chroma",
            "weaviate",
            "pinecone",
            "mongodb (Atlas Vector Search)",
        ]
        vdb_idx = self.p.select("Vector database:", vdb_labels, default_index=0)
        answers["vector_db_type"] = vdb_values[vdb_idx]

        vdb_type = answers["vector_db_type"]
        if vdb_type == "duckdb":
            # Zero-question happy path: defaults are used silently; the user
            # is only consulted when a file/table already exists, so nothing
            # gets overwritten by accident.
            path = "./embeddings.duckdb"
            if Path(path).exists():
                self._section("Vector DB connection")
                path = self.p.text(
                    f"DuckDB file {path} already exists; path to use", default=path
                )
            answers["duckdb_path"] = path
            answers["collection"] = DEFAULT_COLLECTION
            if Path(path).exists() and _duckdb_table_exists(path, DEFAULT_COLLECTION):
                answers["collection"] = self.p.text(
                    f"Table {DEFAULT_COLLECTION!r} already exists in {path}; table to use",
                    default=DEFAULT_COLLECTION,
                )
        elif vdb_type == "milvus":
            self._section("Vector DB connection")
            uri = self.p.text("Milvus URI (leave blank to use host + port)", default="")
            answers["milvus_uri"] = uri or None
            if not answers["milvus_uri"]:
                answers["milvus_host"] = self.p.text("Milvus host", default="localhost")
                answers["milvus_port"] = int(self.p.text("Milvus port", default="19530"))
        elif vdb_type == "qdrant":
            self._section("Vector DB connection")
            answers["qdrant_host"] = self.p.text("Qdrant host", default="localhost")
            answers["qdrant_api_key"] = self.p.text(
                "Qdrant API key (blank for none)", default=""
            ) or None
        elif vdb_type == "chroma":
            self._section("Vector DB connection")
            answers["chroma_host"] = self.p.text("Chroma host", default="localhost")
            answers["chroma_port"] = int(self.p.text("Chroma port", default="8000"))
        elif vdb_type == "weaviate":
            self._section("Vector DB connection")
            answers["weaviate_host"] = self.p.text("Weaviate host", default="localhost")
            answers["weaviate_port"] = int(self.p.text("Weaviate HTTP port", default="8080"))
            answers["weaviate_collection"] = self.p.text(
                "Weaviate collection (capitalized)", default="VetoshEmbeddings"
            )
            answers["weaviate_api_key"] = self.p.text(
                "Weaviate API key (blank for none)", default=""
            ) or None
        elif vdb_type == "pinecone":
            self._section("Vector DB connection")
            answers["pinecone_index"] = self.p.text("Pinecone index name", required=True)
            answers["pinecone_api_key"] = self.p.text(
                "Pinecone API key", default="${PINECONE_API_KEY}"
            )
        elif vdb_type == "mongodb":
            self._section("Vector DB connection")
            answers["mongodb_connection_string"] = self.p.text(
                "MongoDB connection string (mongodb+srv://... for Atlas)", required=True
            )
            answers["mongodb_database"] = self.p.text("Database name", required=True)
        else:
            self._section("Vector DB connection")
            answers["pg_connection_string"] = self.p.text(
                "Postgres connection string (postgresql://user:pass@host/db)", required=True
            )

        if vdb_type not in {"pinecone", "weaviate", "duckdb"}:  # handled above
            self._section("Collection / table name")
            answers["collection"] = self.p.text(
                "Collection / table name", default=DEFAULT_COLLECTION
            )

        self._section("Embedder")
        emb_idx = self.p.select("Embedder:", EMBEDDER_LABELS, default_index=0)
        answers["embedder_type"] = EMBEDDER_CHOICES[emb_idx]

        # 9) A keyless local embedder needs no further questions: the default
        # model applies (changeable later in the YAML).
        env_ref = _ENV_REF.get(answers["embedder_type"])
        if env_ref is not None:
            self._section("Embedder model + API key")
            answers["embedder_model"] = self.p.text("Embedder model (blank for default)", default="") or None
            answers["embedder_api_key"] = self.p.text("API key", default=env_ref)

        # Splitter and persistence are not asked about: defaults are emitted
        # into the YAML for the user to tune there.

        if needs_server:
            # Host/port are silent defaults (0.0.0.0:8989 — uncommon port, no
            # tool conflicts); emitted into the YAML for later tuning.
            self._section("Enable /rag endpoint?")
            answers["rag_enabled"] = self.p.confirm("Enable /rag endpoint?", default=False)
            if answers["rag_enabled"]:
                llm_idx = self.p.select("LLM:", ["openai", "litellm"], default_index=0)
                answers["llm_type"] = ["openai", "litellm"][llm_idx]
                answers["llm_model"] = self.p.text("LLM model", default="gpt-4o-mini")
                answers["llm_api_key"] = self.p.text(
                    "LLM API key", default=_ENV_REF.get(answers["llm_type"], "${OPENAI_API_KEY}")
                )

        self._section("Pathway License Key")
        self.p.info(
            "  Pathway requires a free license key. Get yours in one click at "
            "https://pathway.com/framework/get-license (email or LinkedIn sign-in)."
        )
        answers["license_key"] = self.p.text("Paste your Pathway license key", required=True)

        self._section("Output file path")
        answers["output_path"] = self.p.text("Output file path", default="./config.yaml")

        return answers


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    wizard = Wizard()
    try:
        answers = wizard.run()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        sys.exit(1)

    config = build_config(answers)
    output_path = answers["output_path"]
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(dump_yaml(config))
    print(f"\nWrote {answers['config_type']} config to {output_path}")
    print(
        "Advanced settings (file glob patterns, chunking/splitter, "
        "persistence, backpressure, table names) can be tuned by editing "
        f"{output_path}."
    )
    print(f"Run it with:  {APP_NAME} up --config {output_path}")


if __name__ == "__main__":
    main()
