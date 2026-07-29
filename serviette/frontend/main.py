"""FastAPI web chat frontend (standalone, for split deployments).

A thin, decoupled service: it serves the single-page chat UI and *proxies*
requests to the configured serviette API (``frontend.api_url``). The browser only
ever talks to this frontend's own origin, so there is no CORS to configure and
the API address stays server-side. Use it when the UI must live on a different
host than the API; otherwise ``serviette server`` already serves the same page on
its own port (``server.serve_frontend``, on by default).

The page and the proxy speak the versioned API surface: the browser calls
``/api/v1/...`` on this origin and the proxy forwards to ``/api/v1/...`` on
the upstream server.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from serviette.config.schema import ServietteConfig

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_FILE = _STATIC_DIR / "index.html"

# Endpoints on the upstream API that the frontend is allowed to proxy.
_PROXY_ROUTES = {"rag": "/api/v1/rag", "retrieve": "/api/v1/retrieve"}


def load_index(title: str) -> str:
    """Return the chat page HTML with the configured title substituted.

    Shared with the server's embedded-UI mode, so both serve one page.
    """

    return _INDEX_FILE.read_text(encoding="utf-8").replace("__APP_TITLE__", title)


def create_app(config: ServietteConfig, *, client: httpx.AsyncClient | None = None) -> FastAPI:
    """Build the frontend app.

    ``client`` can be injected (tests provide an httpx client backed by a mock
    transport); otherwise one is created against ``frontend.api_url``.
    """

    config.for_frontend()
    fe = config.frontend
    assert fe is not None  # guaranteed by for_frontend()
    index_html = load_index(fe.title)
    owns_client = client is None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal client
        if client is None:
            client = httpx.AsyncClient(base_url=fe.api_url, timeout=120.0)
        try:
            yield
        finally:
            if owns_client and client is not None:
                await client.aclose()

    app = FastAPI(title=f"{fe.title} frontend", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(index_html)

    @app.get("/api/v1/config")
    async def frontend_config() -> JSONResponse:
        # Informational only (shown in the header); no secrets here.
        return JSONResponse({"title": fe.title, "api_url": fe.api_url})

    async def _proxy(name: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - any malformed body proxies as empty
            body = {}
        assert client is not None  # created in lifespan before serving
        try:
            upstream = await client.post(_PROXY_ROUTES[name], json=body)
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"detail": f"Could not reach the API at {fe.api_url}: {exc}"},
                status_code=502,
            )
        try:
            payload = upstream.json()
        except ValueError:
            payload = {"detail": upstream.text}
        return JSONResponse(payload, status_code=upstream.status_code)

    @app.get("/api/v1/stats")
    async def stats() -> JSONResponse:
        assert client is not None  # created in lifespan before serving
        try:
            upstream = await client.get("/api/v1/stats")
        except httpx.HTTPError:
            return JSONResponse({"stats_available": False}, status_code=502)
        try:
            payload = upstream.json()
        except ValueError:
            payload = {"stats_available": False}
        return JSONResponse(payload, status_code=upstream.status_code)

    @app.post("/api/v1/rag")
    async def rag(request: Request) -> JSONResponse:
        return await _proxy("rag", request)

    @app.post("/api/v1/retrieve")
    async def retrieve(request: Request) -> JSONResponse:
        return await _proxy("retrieve", request)

    return app


def run(config: ServietteConfig) -> None:
    """Start uvicorn with the configured frontend host/port (used by the CLI)."""

    import uvicorn

    config.for_frontend()
    fe = config.frontend
    assert fe is not None  # guaranteed by for_frontend()
    app = create_app(config)
    uvicorn.run(app, host=fe.host, port=fe.port)
