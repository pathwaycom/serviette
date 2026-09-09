"""Second-stage reranking for the server.

Mirrors the embedder module's design: a small async protocol, local
inference moved off the event loop with ``asyncio.to_thread``, and a
``build_reranker`` factory driven by the config section. The cross-encoder
scores every (query, chunk) candidate pair jointly, which recovers matches
that bi-encoder distance ranks poorly; it only ever sees the shortlist the
vector index returns, so cost stays bounded by ``candidates``, not corpus
size.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AsyncReranker(Protocol):
    async def rerank(
        self, query: str, hits: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        ...

    async def close(self) -> None:
        ...


class CrossEncoderReranker:
    """Rerank with a local sentence-transformers cross-encoder.

    ``is_local = True`` opts into the server's startup warm-up, same as the
    local embedder: the first user question must not pay the model load.
    """

    _DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    is_local = True

    def __init__(self, config) -> None:
        self._model_name = config.model or self._DEFAULT_MODEL
        extra = config.model_dump(exclude={"type", "model", "candidates"})
        self._model_kwargs = {k: v for k, v in extra.items() if v is not None}
        self._model_kwargs.setdefault("device", "cpu")
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, **self._model_kwargs)
        return self._model

    async def rerank(
        self, query: str, hits: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        import asyncio

        if not hits:
            return hits
        model = await asyncio.to_thread(self._ensure_model)
        scores = await asyncio.to_thread(
            model.predict, [(query, hit["text"]) for hit in hits]
        )
        reranked = [
            {**hit, "score": float(score)}
            for score, hit in sorted(
                zip(scores, hits), key=lambda pair: -float(pair[0])
            )
        ]
        return reranked[:k]

    async def close(self) -> None:
        self._model = None


_RERANK_PROMPT = """\
Rate how relevant the document is for answering the query, as an integer \
from 0 (irrelevant) to 5 (directly answers it). Respond with the number only.

Query: {query}

Document: {doc}"""


class LLMReranker:
    """Pointwise LLM reranking, after ``pathway.xpacks.llm.rerankers.LLMReranker``.

    Each candidate is rated 0–5 by an LLM (one cheap call per candidate,
    issued concurrently). An unparseable rating scores the candidate 0 rather
    than failing the request. ``is_local = False``: nothing to warm up.
    """

    is_local = False

    def __init__(self, config, llm_config=None, *, chat=None) -> None:
        # The scorer is any AsyncLLM (only ``raw`` is used). Tests inject a
        # fake via ``chat``; production lazily builds the configured backend
        # (build_llm) from the reranker section, falling back to the
        # top-level llm section.
        self._chat = chat
        self._config = config
        self._llm_config = llm_config

    def _ensure_chat(self):
        if self._chat is None:
            merged = dict(self._llm_config.model_dump() if self._llm_config else {})
            overrides = self._config.model_dump(exclude={"type", "candidates"})
            merged.update({k: v for k, v in overrides.items() if v is not None})
            merged.setdefault("type", "openai")

            from serviette.config.schema import LLMConfig
            from serviette.server.llm import build_llm

            # build_llm, not OpenAIChat directly: a litellm-typed llm section
            # must route through LiteLLM, not the plain OpenAI client.
            self._chat = build_llm(LLMConfig(**merged))
        return self._chat

    async def rerank(
        self, query: str, hits: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        import asyncio

        if not hits:
            return hits
        chat = self._ensure_chat()

        async def score(hit: dict[str, Any]) -> float:
            reply = await chat.raw(
                _RERANK_PROMPT.format(query=query, doc=hit["text"])
            )
            return _parse_rating(reply)

        scores = await asyncio.gather(*(score(hit) for hit in hits))
        reranked = [
            {**hit, "score": float(s)}
            for s, hit in sorted(zip(scores, hits), key=lambda pair: -pair[0])
        ]
        return reranked[:k]

    async def close(self) -> None:
        if self._chat is not None:
            await self._chat.close()
            self._chat = None


def _parse_rating(reply: str) -> float:
    """First number in the reply, or 0.0 when there is none."""

    import re

    match = re.search(r"\d+(?:\.\d+)?", reply)
    return float(match.group()) if match else 0.0


def build_reranker(config, llm_config=None) -> AsyncReranker:
    if config.type in {"cross_encoder", "crossencoder"}:
        return CrossEncoderReranker(config)
    if config.type == "llm":
        return LLMReranker(config, llm_config)
    raise ValueError(f"Unsupported reranker type: {config.type!r}")
