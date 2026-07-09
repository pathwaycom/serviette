"""FastAPI retrieval/RAG server.

Async and coroutine-based so network-bound calls to the vector DB and embedding
provider don't exhaust a thread pool. Fully decoupled from the indexer: it only
reads from the vector DB, so it can be scaled horizontally and independently.

API surface
-----------
Endpoints are versioned under ``/api/v1`` (``/api/v1/retrieve``,
``/api/v1/rag``, ``/api/v1/health``, ``/api/v1/config``). When the API evolves
incompatibly, a ``/api/v2`` router is added next to v1 and v1 sticks around
for a deprecation window. The original unversioned routes (``/retrieve``,
``/rag``, ``/health``) are kept as deprecated aliases for compatibility.

Unless ``server.serve_frontend`` is disabled, the chat UI is served on ``/``
from this same process/port — same-origin, so no CORS and no separate
frontend tier to run. The standalone ``vetosh frontend`` command remains for
split deployments (UI host separate from the API host).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from vetosh import APP_NAME
from vetosh.config.schema import VetoshConfig
from vetosh.server.accessors import AsyncVectorAccessor, build_accessor
from vetosh.server.accessors.abstract import IndexNotReadyError
from vetosh.server.embedder import AsyncEmbedder, build_embedder
from vetosh.server.llm import AsyncLLM, build_llm

logger = logging.getLogger(__name__)


class RetrieveRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1)


class RetrieveResult(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float


class RetrieveResponse(BaseModel):
    results: list[RetrieveResult]


class RagResponse(BaseModel):
    answer: str
    sources: list[RetrieveResult]


def create_app(
    config: VetoshConfig,
    *,
    embedder: AsyncEmbedder | None = None,
    accessor: AsyncVectorAccessor | None = None,
    llm: AsyncLLM | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    The ``embedder``/``accessor``/``llm`` overrides exist for testing (inject a
    mock embedder and a DuckDB accessor); in production they are built from the
    config.
    """

    config.for_server()

    embedder = embedder or build_embedder(config.embedder)
    accessor = accessor or build_accessor(config.vector_db)
    # ``/rag`` is enabled only when an LLM is configured (or injected).
    if llm is None and config.llm is not None:
        llm = build_llm(config.llm)

    title = config.frontend.title if config.frontend else APP_NAME

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Local embedders (sentence-transformers) lazily import torch and
        # load weights on first use — seconds the first user question must
        # not pay. Warm them up with an invisible query before serving;
        # API embedders skip this (a warm-up there costs real tokens).
        if getattr(embedder, "is_local", False):
            import time as _time

            started = _time.monotonic()
            logger.info("warming up the local embedder...")
            await embedder.embed("vetosh warmup")
            logger.info(
                "embedder ready in %.1fs", _time.monotonic() - started
            )
        try:
            yield
        finally:
            await accessor.close()
            await embedder.close()
            if llm is not None:
                await llm.close()

    app = FastAPI(title=f"{title} server", lifespan=lifespan)

    if config.server and config.server.cors_origins:
        # Off by default on purpose (see ServerConfig.cors_origins); an
        # explicit allowlist opts in third-party browser frontends.
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(IndexNotReadyError)
    async def _index_not_ready(_request, exc: IndexNotReadyError):
        # The indexer simply hasn't written its first batch yet — a normal
        # state during startup, not an error worth a stack trace.
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "The index is not ready yet — the indexer is still "
                    "starting or hasn't written its first documents. "
                    "Try again in a moment."
                ),
                "reason": str(exc),
            },
            headers={"Retry-After": "5"},
        )
    v1 = APIRouter(prefix="/api/v1")

    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
        embedding = await embedder.embed(req.query)
        hits = await accessor.retrieve(embedding, req.k)
        return RetrieveResponse(results=[RetrieveResult(**h) for h in hits])

    async def rag(req: RetrieveRequest) -> RagResponse:
        if llm is None:
            raise HTTPException(
                status_code=501,
                detail="The /rag endpoint requires an 'llm' config section.",
            )
        embedding = await embedder.embed(req.query)
        hits = await accessor.retrieve(embedding, req.k)
        answer = await llm.complete(req.query, [h["text"] for h in hits])
        return RagResponse(
            answer=answer, sources=[RetrieveResult(**h) for h in hits]
        )

    async def stats() -> dict[str, Any]:
        # Best-effort observability: backend identity + whatever the accessor
        # can answer cheaply. Never fails the endpoint over a backend hiccup.
        data: dict[str, Any] = {"backend": config.vector_db.type}
        try:
            data.update(await accessor.stats())
        except Exception:  # noqa: BLE001 - stats are advisory
            data["stats_available"] = False
        return data

    async def ui_config() -> dict[str, str]:
        # Informational, consumed by the chat page; an empty api_url means
        # "same origin" (embedded mode). No secrets here.
        return {"title": title, "api_url": ""}

    v1.get("/health")(health)
    v1.post("/retrieve", response_model=RetrieveResponse)(retrieve)
    v1.post("/rag", response_model=RagResponse)(rag)
    v1.get("/stats")(stats)
    v1.get("/config")(ui_config)
    app.include_router(v1)

    # Pre-versioning aliases, kept for compatibility; will be removed after a
    # deprecation window. New clients must use /api/v1/*.
    app.get("/health", deprecated=True)(health)
    app.post("/retrieve", response_model=RetrieveResponse, deprecated=True)(retrieve)
    app.post("/rag", response_model=RagResponse, deprecated=True)(rag)

    if config.server.serve_frontend:
        from vetosh.frontend.main import load_index

        index_html = load_index(title)

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def index() -> HTMLResponse:
            return HTMLResponse(index_html)

    return app


def run(config: VetoshConfig) -> None:
    """Start uvicorn with the configured host/port (used by the CLI)."""

    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)
