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
frontend tier to run. The standalone ``serviette frontend`` command remains for
split deployments (UI host separate from the API host).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from serviette import APP_NAME
from serviette.config.schema import (
    AdaptiveRagConfig,
    DocumentsConfig,
    ServietteConfig,
)
from serviette.server import documents as docmode
from serviette.server.accessors import AsyncVectorAccessor, build_accessor
from serviette.server.accessors.abstract import IndexBusyError, IndexNotReadyError
from serviette.server.decompose import decompose_query
from serviette.server.embedder import AsyncEmbedder, build_embedder
from serviette.server.llm import DEFAULT_SYSTEM_PROMPT, AsyncLLM, build_llm
from serviette.server.ranking import interleave_merge, mmr_select
from serviette.server.reranker import AsyncReranker, build_reranker

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


class DocumentEntry(BaseModel):
    id: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunks: int


class DocumentsResponse(BaseModel):
    """``GET /api/v1/documents``: the indexed documents whose name or path
    contains ``q``. The list is left out when more than ``limit`` match —
    the caller narrows the search instead of rendering thousands of rows."""

    total: int  # documents in the index
    matched: int  # of them, matching the filter
    truncated: bool  # matched > limit: ``documents`` was not sent
    documents: list[DocumentEntry] = Field(default_factory=list)


class RagResponse(BaseModel):
    answer: str
    sources: list[RetrieveResult]
    # How the answer was produced: "search" (chunk retrieval), "catalog"
    # (the document listing), "document" / "compare" (whole named documents)
    # or "unresolved" (the question named documents that could not be
    # identified). Anything but "search" comes from ``rag.documents``.
    mode: str = "search"
    # The documents the answer is scoped to (document / compare modes).
    documents: list[str] = Field(default_factory=list)
    # A system-computed remark on the answer's coverage, e.g. that only part
    # of a document was read or how many differences were found.
    notice: str | None = None


class _IndexChangeTracker:
    """Notice index changes the stored timestamps cannot show.

    Accessors derive ``last_indexed_at`` from the newest row's ``seen_at``,
    so a deletion (or an edit that only removes chunks) silently rolls the
    value back to an older document. The tracker compares successive stats
    snapshots and, when the row count or newest timestamp moves, records the
    wall-clock time of that observation; ``observe`` then reports whichever
    is later. State is per server process: after a restart the backend's
    own value is used until the next change.
    """

    def __init__(self) -> None:
        self._snapshot: tuple | None = None
        self._changed_at: int | None = None

    def observe(self, stats: dict[str, Any]) -> dict[str, Any]:
        import time

        key = (
            stats.get("chunks"),
            stats.get("documents"),
            stats.get("last_indexed_at"),
        )
        if self._snapshot is None:
            self._snapshot = key  # first look: nothing to compare against
        elif key != self._snapshot:
            self._snapshot = key
            self._changed_at = int(time.time())
        stored = stats.get("last_indexed_at")
        if self._changed_at is not None and (
            stored is None or self._changed_at > stored
        ):
            stats["last_indexed_at"] = self._changed_at
        return stats


# Appended to a response's ``notice`` when the request ran into
# rag.documents.max_request_chars.
_BUDGET_NOTICE = (
    " The request reached its LLM budget (rag.documents.max_request_chars), "
    "so part of the material was left out."
)

# How often the server itself polls the backend for changes, so deletions
# are stamped even while no chat page is open. Matches the page's own poll.
_INDEX_POLL_SECONDS = 5.0


def create_app(
    config: ServietteConfig,
    *,
    embedder: AsyncEmbedder | None = None,
    accessor: AsyncVectorAccessor | None = None,
    llm: AsyncLLM | None = None,
    reranker: AsyncReranker | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    The ``embedder``/``accessor``/``llm``/``reranker`` overrides exist for
    testing (inject a mock embedder and a DuckDB accessor); in production they
    are built from the config.
    """

    config.for_server()
    # for_server() guarantees the sections; narrow them for the type checker.
    assert config.vector_db is not None and config.embedder is not None

    embedder = embedder or build_embedder(config.embedder)
    accessor = accessor or build_accessor(config.vector_db)
    # ``/rag`` is enabled only when an LLM is configured (or injected).
    if llm is None and config.llm is not None:
        llm = build_llm(config.llm)
    # Reranking is opt-in: without a ``reranker`` section the vector-index
    # order is returned as-is.
    if reranker is None and config.reranker is not None:
        reranker = build_reranker(config.reranker, config.llm)

    # Retrieval-quality strategies (rag section) — validate at startup, not
    # on the first unlucky request.
    rag_cfg = config.rag
    adaptive = rag_cfg.adaptive if rag_cfg else None
    decompose = rag_cfg.decompose if rag_cfg else None
    mmr = rag_cfg.mmr if rag_cfg else None
    if decompose is not None and llm is None:
        raise ValueError(
            "rag.decompose requires an 'llm' config section (it uses one "
            "LLM call to split the question into sub-queries)."
        )
    if mmr is not None and not accessor.supports_embeddings:
        raise ValueError(
            "rag.mmr needs hit embeddings, which the "
            f"'{config.vector_db.type}' backend accessor does not return."
        )
    # On by default, so a backend that cannot enumerate documents turns it
    # off quietly instead of refusing to start.
    documents = rag_cfg.documents if rag_cfg else DocumentsConfig()
    document_mode = documents.enabled and accessor.supports_catalog
    if documents.enabled and not accessor.supports_catalog:
        logger.info(
            "rag.documents is off: the '%s' backend cannot list its documents",
            config.vector_db.type,
        )
    # The reply that marks an attempt as failed: it drives the adaptive loop
    # and, in document mode, the switch from search to documents.
    no_answer = (
        adaptive.no_answer_string
        if adaptive
        else AdaptiveRagConfig().no_answer_string
    )

    title = config.frontend.title if config.frontend else APP_NAME
    tracker = _IndexChangeTracker()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Local embedders (sentence-transformers) lazily import torch and
        # load weights on first use — seconds the first user question must
        # not pay. Warm them up with an invisible query before serving;
        # API embedders skip this (a warm-up there costs real tokens).
        if getattr(embedder, "is_local", False):
            import time as _time

            started = _time.monotonic()
            logger.info(
                "warming up the local embedder (importing the ML stack and "
                "loading the model — the port opens once this is done)..."
            )
            await embedder.embed("serviette warmup")
            logger.info(
                "embedder ready in %.1fs", _time.monotonic() - started
            )
        if reranker is not None and getattr(reranker, "is_local", False):
            import time as _time

            started = _time.monotonic()
            logger.info("warming up the local reranker...")
            await reranker.rerank("serviette warmup", [{"text": "warmup"}], 1)
            logger.info(
                "reranker ready in %.1fs", _time.monotonic() - started
            )
        # API clients (LLM, LLM reranker) are built lazily; building them
        # here moves the SDK import and connection setup out of the first
        # /rag request. No request is made — nothing billable.
        for name, component in (("llm", llm), ("reranker", reranker)):
            prepare = getattr(component, "prepare", None)
            if prepare is None:
                continue
            import time as _time

            started = _time.monotonic()
            await prepare()
            logger.info(
                "%s client ready in %.1fs", name, _time.monotonic() - started
            )
        import asyncio

        async def _poll_index() -> None:
            # Failures here are the backend being not-ready or briefly
            # locked; the next tick simply looks again.
            while True:
                try:
                    tracker.observe(await accessor.stats())
                except Exception as exc:  # noqa: BLE001 - advisory only
                    logger.debug("index poll skipped: %s", exc)
                await asyncio.sleep(_INDEX_POLL_SECONDS)

        poller = asyncio.create_task(_poll_index())
        try:
            yield
        finally:
            poller.cancel()
            await accessor.close()
            await embedder.close()
            if llm is not None:
                await llm.close()
            if reranker is not None:
                await reranker.close()

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
        # The indexer simply hasn't written its first batch yet, or is in
        # the middle of writing one — normal states, not errors worth a
        # stack trace.
        from fastapi.responses import JSONResponse

        if isinstance(exc, IndexBusyError):
            detail = (
                "The index is being updated — the indexer is writing a batch "
                "of documents right now. Try again in a few seconds."
            )
        else:
            detail = (
                "The index is not ready yet — the indexer is still "
                "starting or hasn't written its first documents. "
                "Try again in a moment."
            )
        return JSONResponse(
            status_code=503,
            content={"detail": detail, "reason": str(exc)},
            headers={"Retry-After": "5"},
        )
    v1 = APIRouter(prefix="/api/v1")

    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # Asymmetric-retrieval models (e5, bge) expect a query-side marker; the
    # indexer applies the matching document_prefix. Empty for symmetric models.
    backend_type = config.vector_db.type
    query_prefix = config.embedder.query_prefix
    rerank_candidates = config.reranker.candidates if config.reranker else 0

    async def _queries(query: str) -> list[str]:
        """The retrieval queries for ``query``: itself, plus its LLM
        sub-queries when ``rag.decompose`` is on. Computed once per request —
        the adaptive loop re-searches with a growing ``k`` and must not pay
        (or re-roll) the decomposition on every round."""

        if decompose is None:
            return [query]
        assert llm is not None  # validated at startup when decompose is set
        return await decompose_query(llm, query, decompose.max_subqueries)

    async def _search(
        query: str, k: int, queries: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Retrieval pipeline: (decompose) → fetch pool → (rerank) → (MMR) → top-k.

        Each optional stage is driven by its config section; with none of
        them configured this reduces to the plain embed-and-retrieve path.
        ``queries`` lets a caller that searches repeatedly (adaptive RAG)
        pass the decomposition it already obtained.
        """

        # A reranker or MMR selects k out of a wider candidate pool.
        pool = max(k, rerank_candidates, mmr.candidates if mmr else 0)
        need_embeddings = mmr is not None

        if queries is None:
            queries = await _queries(query)

        async def fetch(q: str) -> list[dict[str, Any]]:
            embedding = await embedder.embed(query_prefix + q)
            return await accessor.retrieve_ex(
                embedding, pool, query_text=q, with_embeddings=need_embeddings
            )

        if len(queries) == 1:
            hits = await fetch(query)
        else:
            import asyncio

            per_query = await asyncio.gather(*(fetch(q) for q in queries))
            # Round-robin, not summed RRF: each sub-query's best chunk is
            # guaranteed a slot (see interleave_merge).
            hits = interleave_merge(list(per_query), pool)

        if reranker is not None:
            # Keep the pool wide when MMR still has to diversify after us.
            keep = max(k, mmr.candidates) if mmr else k
            hits = await reranker.rerank(query, hits, keep)
        if mmr is not None:
            hits = mmr_select(hits, k, mmr.diversity)
        hits = hits[:k]
        # Embeddings are pipeline plumbing, not API surface.
        return [{key: v for key, v in h.items() if key != "embedding"} for h in hits]

    async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
        hits = await _search(req.query, req.k)
        return RetrieveResponse(results=[RetrieveResult(**h) for h in hits])

    async def rag(req: RetrieveRequest) -> RagResponse:
        if llm is None:
            raise HTTPException(
                status_code=501,
                detail="The /rag endpoint requires an 'llm' config section.",
            )
        catalog = await _catalog() if document_mode else []
        # One ceiling for everything this request sends to the LLM, across
        # all of its calls (see docmode.Budget).
        budget = docmode.Budget(documents.max_request_chars) if catalog else None
        if catalog:
            # A file named outright needs no search to find it — and a search
            # would not find it anyway: chunk texts do not carry file names.
            named = docmode.mentioned_names(req.query, catalog)
            if named:
                return await _answer_documents(req.query, catalog, named, budget)
        if adaptive is None and not catalog:
            hits = await _search(req.query, req.k)
            answer = await llm.complete(req.query, [h["text"] for h in hits])
            return RagResponse(
                answer=answer, sources=[RetrieveResult(**h) for h in hits]
            )
        # The system prompt is independent parts: the answering *policy* (the
        # configured llm.system_prompt, or the built-in grounded default),
        # in document mode the corpus card, and the no-answer *protocol* —
        # always appended, since the marker is what tells the loop below that
        # the attempt failed.
        policy = getattr(config.llm, "system_prompt", None) or DEFAULT_SYSTEM_PROMPT
        card = (
            f"\n\n{docmode.corpus_card(catalog)}"
            if catalog and documents.corpus_card
            else ""
        )
        label_sources = bool(catalog) and documents.source_labels
        if label_sources:
            card += "\n\n" + docmode.SOURCE_LABELS_NOTE.format(marker=no_answer)
        system_prompt = (
            f"{policy}{card}\n\n"
            "If the provided context does not contain the information needed "
            f'to answer, reply exactly "{no_answer}".'
        )
        # Adaptive RAG: grow the context geometrically while the LLM reports
        # that it cannot answer from what it was given. Without rag.adaptive
        # there is a single attempt.
        k = req.k
        queries = await _queries(req.query)
        arbitrated = False
        answer, hits = no_answer, []
        for _iteration in range(adaptive.max_iterations if adaptive else 1):
            found = await _search(req.query, k, queries)
            context = (
                docmode.labeled(found) if label_sources else [h["text"] for h in found]
            )
            attempt = await _complete(budget, req.query, context, system_prompt)
            if attempt is None:
                break  # out of budget: keep the previous attempt's outcome
            answer, hits = attempt[0], found[: len(attempt[1])]
            if no_answer not in answer:
                break
            if catalog and not arbitrated:
                # First failure: one short call decides who continues — the
                # loop (a wider search may still find it) or document mode
                # (no amount of chunks answers a question about documents).
                # The switch is one-way and happens at most once.
                arbitrated = True
                route = await docmode.arbitrate(
                    llm, req.query, catalog, documents.max_listed_documents, budget
                )
                if route.mode == "catalog":
                    return await _answer_catalog(req.query, catalog, budget)
                if route.mode == "documents":
                    return await _answer_documents(
                        req.query, catalog, route.files, budget
                    )
            if adaptive is None:
                break
            if len(found) < k:
                # Retrieval returned less than asked: the corpus (or the
                # candidate pool) is exhausted, so a larger k would hand the
                # LLM the very same context again. Stop instead of repeating
                # the identical call until max_iterations.
                break
            k *= adaptive.factor
        notice = None
        if catalog and no_answer in answer:
            notice = (
                f"No answer was found in the {len(catalog)} indexed documents."
            )
            if budget is not None and budget.exhausted:
                notice += _BUDGET_NOTICE
        return RagResponse(
            answer=answer,
            sources=[RetrieveResult(**h) for h in hits],
            notice=notice,
        )

    async def _complete(
        budget: docmode.Budget | None,
        query: str,
        context: list[str],
        system_prompt: str | None,
    ) -> tuple[str, list[str]] | None:
        """One answering call within the request's budget: the reply and the
        context it was actually given (trailing items are dropped when the
        budget is short), or ``None`` when nothing useful fits any more."""

        assert llm is not None
        if budget is not None:
            fitted = budget.fit(len(system_prompt or "") + len(query), context)
            if fitted is None:
                return None
            context = fitted
        reply = await llm.complete(query, context, system_prompt=system_prompt)
        return reply, context

    async def _catalog() -> list[dict[str, Any]]:
        """The document catalog, or ``[]`` when it cannot be had right now —
        document mode is an addition and must never fail the ordinary path
        (which reports a not-ready index properly on its own)."""

        try:
            return await accessor.list_documents()
        except Exception as exc:  # noqa: BLE001 - fall back to plain search
            logger.debug("document catalog unavailable: %s", exc)
            return []

    async def _answer_catalog(
        query: str, catalog: list[dict[str, Any]], budget: docmode.Budget | None
    ) -> RagResponse:
        context = docmode.catalog_context(catalog, documents.max_context_chars)
        reply = await _complete(
            budget, query, context, docmode.CATALOG_SYSTEM_PROMPT
        )
        if reply is None:
            # Out of budget: the totals are the system's own and stand alone.
            return RagResponse(
                answer=context[0], sources=[], mode="catalog", notice=_BUDGET_NOTICE.strip()
            )
        return RagResponse(answer=reply[0], sources=[], mode="catalog")

    async def _unresolved(
        query: str, facts: str, budget: docmode.Budget | None
    ) -> RagResponse:
        """The question cannot be answered as asked: say why, concretely.
        The facts are computed here; the LLM only words them in the user's
        language, and they stand on their own if it fails to."""

        try:
            reply = await _complete(
                budget, query, [facts], docmode.EXPLAIN_SYSTEM_PROMPT
            )
        except Exception:  # noqa: BLE001 - the facts are the answer then
            logger.warning("could not word a document-mode error; sending facts")
            reply = None
        answer = reply[0] if reply else ""
        return RagResponse(
            answer=answer.strip() or facts,
            sources=[],
            mode="unresolved",
            notice=facts,
        )

    async def _answer_documents(
        query: str,
        catalog: list[dict[str, Any]],
        names: list[str],
        budget: docmode.Budget | None,
    ) -> RagResponse:
        """Answer from whole named documents: one is read (in full, or its
        parts closest to the question), several are compared."""

        assert llm is not None
        resolution = docmode.resolve_names(names, catalog)
        problems = [
            f'No indexed document is named "{name}".'
            + (f" Similar names: {', '.join(similar)}." if similar else "")
            for name, similar in resolution.missing.items()
        ]
        if resolution.missing and not any(resolution.missing.values()):
            # Nothing close to offer: show what there is instead.
            names_known = sorted(docmode.document_name(e) for e in catalog)
            shown = ", ".join(names_known[:20])
            more = len(names_known) - 20
            problems.append(
                f"Indexed documents: {shown}"
                + (f" and {more} more." if more > 0 else ".")
            )
        problems += [
            f'"{name}" matches several indexed documents: {", ".join(ids)}. '
            "Use the full path to say which one."
            for name, ids in resolution.ambiguous.items()
        ]
        if not problems and not resolution.found:
            problems = [
                (
                    "The question is about specific documents but does not "
                    f"name them. There are {len(catalog)} indexed documents; "
                    "name the files to use."
                )
            ]
        if problems:
            return await _unresolved(query, " ".join(problems), budget)

        entries = resolution.found
        chunks = [
            await accessor.document_chunks(entry["id"], with_embeddings=True)
            for entry in entries
        ]
        gone = [
            docmode.document_name(entry)
            for entry, doc_chunks in zip(entries, chunks)
            if not doc_chunks
        ]
        if gone:
            return await _unresolved(
                query, f"No longer in the index: {', '.join(gone)}.", budget
            )
        query_embedding = await embedder.embed(query_prefix + query)
        limit = documents.max_context_chars
        if len(entries) == 1:
            mode = "document"
            built = docmode.single_document_context(
                entries[0], chunks[0], query_embedding, limit
            )
        else:
            import asyncio

            mode = "compare"
            # Pairing changed passages is CPU-bound; keep it off the loop.
            built = await asyncio.to_thread(
                docmode.compare_context, entries, chunks, query_embedding, limit
            )
        reply = await _complete(budget, query, built.context, built.system_prompt)
        notice = built.notice
        if reply is None or len(reply[1]) < len(built.context):
            notice = (notice or "") + _BUDGET_NOTICE
        return RagResponse(
            # Out of budget: the header (what was read, the exact totals) is
            # the system's own and stands alone.
            answer=reply[0] if reply else built.context[0],
            sources=[RetrieveResult(**h) for h in built.sources],
            mode=mode,
            documents=[str(entry["id"]) for entry in entries],
            notice=notice.strip() if notice else None,
        )

    async def stats() -> dict[str, Any]:
        # Best-effort observability: backend identity + whatever the accessor
        # can answer cheaply. Never fails the endpoint over a backend hiccup.
        data: dict[str, Any] = {"backend": backend_type}
        try:
            data.update(tracker.observe(await accessor.stats()))
        except Exception:  # noqa: BLE001 - stats are advisory
            data["stats_available"] = False
        return data

    async def list_documents(
        q: str = "", case_sensitive: bool = False, limit: int = 100
    ) -> DocumentsResponse:
        """The document catalog, filtered by a fragment of the name or path.

        Served from the accessor's cached catalog (nothing is scanned per
        request), and only on demand — the chat page asks when the person
        opens its document panel and searches, never on load.
        """

        if not accessor.supports_catalog:
            raise HTTPException(
                status_code=501,
                detail=f"The '{backend_type}' backend cannot list its documents.",
            )
        limit = max(0, min(limit, 1000))
        catalog = await accessor.list_documents()
        needle = q if case_sensitive else q.casefold()

        def matches(entry: dict[str, Any]) -> bool:
            if not needle:
                return True
            haystacks = (docmode.document_name(entry), str(entry["id"]))
            return any(
                (needle in h) if case_sensitive else (needle in h.casefold())
                for h in haystacks
            )

        matched = [entry for entry in catalog if matches(entry)]
        truncated = len(matched) > limit
        documents = (
            []
            if truncated
            else [
                DocumentEntry(
                    id=str(e["id"]),
                    name=docmode.document_name(e),
                    metadata=e.get("metadata") or {},
                    chunks=int(e.get("chunks", 0)),
                )
                for e in sorted(matched, key=docmode.document_name)
            ]
        )
        return DocumentsResponse(
            total=len(catalog), matched=len(matched), truncated=truncated, documents=documents
        )

    async def ui_config() -> dict[str, str]:
        # Informational, consumed by the chat page; an empty api_url means
        # "same origin" (embedded mode). No secrets here.
        return {"title": title, "api_url": ""}

    v1.get("/health")(health)
    v1.post("/retrieve", response_model=RetrieveResponse)(retrieve)
    v1.post("/rag", response_model=RagResponse)(rag)
    v1.get("/stats")(stats)
    v1.get("/documents", response_model=DocumentsResponse)(list_documents)
    v1.get("/config")(ui_config)
    app.include_router(v1)

    # Pre-versioning aliases, kept for compatibility; will be removed after a
    # deprecation window. New clients must use /api/v1/*.
    app.get("/health", deprecated=True)(health)
    app.post("/retrieve", response_model=RetrieveResponse, deprecated=True)(retrieve)
    app.post("/rag", response_model=RagResponse, deprecated=True)(rag)

    if config.server.serve_frontend:
        from serviette.frontend.main import load_index

        index_html = load_index(title)

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def index() -> HTMLResponse:
            return HTMLResponse(index_html)

    return app


def run(config: ServietteConfig) -> None:
    """Start uvicorn with the configured host/port (used by the CLI)."""

    import uvicorn

    from serviette.config.schema import require_openai_credentials

    # The embedder is needed on the first /retrieve and the LLM on the first
    # /rag: without a key both would fail as a 500 then. Say it now instead.
    require_openai_credentials(config, roles=("embedder", "llm"))

    # uvicorn configures only its own loggers; without a root handler the
    # warm-up progress above is silently dropped and the user stares at
    # "Waiting for application startup." for as long as the model loads.
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Chatty at INFO: one line per HTTP request (httpx) and per model file
    # the hub checks (sentence-transformers / huggingface_hub).
    for name in ("httpx", "httpcore", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)
