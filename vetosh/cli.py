"""Single ``vetosh`` entrypoint dispatching all subcommands.

Subcommands:
  vetosh indexer    --config FILE   Build & run the Pathway indexing graph
  vetosh server     --config FILE   Start the FastAPI retrieval/RAG server
  vetosh up         --config FILE   Run indexer + server together (dev/demo)
  vetosh quickstart                 Interactive config-generation wizard
"""

from __future__ import annotations

import argparse
import sys

from vetosh import APP_NAME, __version__


_DEV_PATHWAY_HINT = (
    "Your installed pathway ({version}) is a released build without the "
    "vector-store connectors vetosh needs (they have not shipped in a "
    "release yet). Install the development build into this environment:\n\n"
    "  uv pip install -U pathway --prerelease=allow \\\n"
    "      --extra-index-url https://packages.pathway.com/966431ef6ba\n"
)


def _ensure_dev_pathway() -> None:
    """Fail with instructions, not an AttributeError, on a released pathway.

    Engine-facing commands (indexer/up/demo) need connectors that exist only
    in development builds; a plain `pip install pathway` from PyPI resolves
    to a release without them, and the first symptom used to be an opaque
    crash deep in the stack.
    """

    import pathway as pw

    if not hasattr(pw.io, "duckdb"):
        raise SystemExit(_DEV_PATHWAY_HINT.format(version=pw.__version__))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(prog=APP_NAME, description=__doc__)
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("indexer", add_help=False, help="Run the indexer")
    sub.add_parser("server", add_help=False, help="Run the server")
    sub.add_parser("up", add_help=False, help="Run indexer + server together")
    sub.add_parser("demo", add_help=False, help="Zero-to-chat demo on a bundled corpus")
    sub.add_parser("frontend", add_help=False, help="Run the web chat frontend")
    sub.add_parser("quickstart", add_help=False, help="Interactive config wizard")

    # Parse only the first token so each subcommand owns the rest of argv.
    args, rest = parser.parse_known_args(argv)

    if args.command in ("indexer", "up", "demo"):
        _ensure_dev_pathway()
    if args.command == "indexer":
        from vetosh.indexer.main import main as indexer_main

        indexer_main(rest)
    elif args.command == "server":
        from vetosh.server.main import run
        from vetosh.server.config import load_server_config

        server_args = _parse_config_arg("server", rest)
        run(load_server_config(server_args.config))
    elif args.command == "up":
        from vetosh.up import main as up_main

        up_main(rest)
    elif args.command == "demo":
        from vetosh.demo import main as demo_main

        demo_main(rest)
    elif args.command == "frontend":
        from vetosh.frontend.config import load_frontend_config
        from vetosh.frontend.main import run

        frontend_args = _parse_config_arg("frontend", rest)
        run(load_frontend_config(frontend_args.config))
    elif args.command == "quickstart":
        from vetosh.quickstart.wizard import main as quickstart_main

        quickstart_main(rest)


def _parse_config_arg(prog: str, rest: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=f"{APP_NAME} {prog}")
    p.add_argument("--config", required=True, help="Path to the YAML config file")
    return p.parse_args(rest)


if __name__ == "__main__":
    main()
