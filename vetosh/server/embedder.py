"""Query embedding for the server.

Design note — why not call the xpack embedder directly?
-------------------------------------------------------
``pathway.xpacks.llm.embedders`` classes are Pathway *UDFs*: calling them builds
graph nodes, not plain values, and they're meant to run inside ``pw.run``. The
server is a network-bound async FastAPS service and is deliberately decoupled
from Pathway (it need not even have ``pathway`` installed). So we embed a single
query by calling the provider SDK directly as a coroutine — exactly the
"coroutines avoid thread-pool exhaustion" rationale for choosing FastAPI.

The ``type``/``model`` config keys mirror the indexer's embedder config, so the
**same** embedder section produces matching vectors on both sides as long as the
model matches. Tests inject a deterministic mock implementing
:class:`AsyncEmbedder`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AsyncEmbedder(Protocol):
    async def embed(self, text: str) -> list[float]:
        ...

    async def close(self) -> None:
        ...


class MockAsyncEmbedder:
    """Deterministic, offline embedder (dev/test scaffold).

    Uses the same hash-based ``fake_embedding`` as the indexer's ``mock``
    embedder, so query and document vectors are produced identically and the
    whole stack runs with no provider or credentials. Not for real semantic
    search — switch to a real embedder for that.
    """

    async def embed(self, text: str) -> list[float]:
        from vetosh.testing import fake_embedding

        return fake_embedding(text)

    async def close(self) -> None:
        return None


class OpenAIAsyncEmbedder:
    """Embed queries with OpenAI (or any OpenAI-compatible endpoint)."""

    def __init__(self, config) -> None:
        self._model = config.model or "text-embedding-3-small"
        self._api_key = config.api_key
        # Forward any extra keys (e.g. base_url) declared on the config.
        extra = config.model_dump(exclude={"type", "model", "api_key"})
        self._client_kwargs = {k: v for k, v in extra.items() if v is not None}
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key, **self._client_kwargs)
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._ensure_client()
        resp = await client.embeddings.create(model=self._model, input=[text])
        return list(resp.data[0].embedding)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


class SentenceTransformerAsyncEmbedder:
    """Embed queries with a local sentence-transformers model.

    Fully local — no provider, no credentials. The (CPU/GPU-bound) encode runs
    off the event loop in a worker thread. Mirrors the indexer's
    ``sentence_transformer`` xpack embedder; use the same ``model`` on both
    sides.
    """

    _DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, config) -> None:
        self._model_name = config.model or self._DEFAULT_MODEL
        # Forward extra config keys (device, truncate_dim, ...) to the
        # SentenceTransformer constructor — mirrors the indexer's xpack
        # embedder, so e.g. Matryoshka truncation stays consistent.
        extra = config.model_dump(exclude={"type", "model", "api_key", "batch_size"})
        self._model_kwargs = {k: v for k, v in extra.items() if v is not None}
        self._model_kwargs.setdefault("device", "cpu")
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, **self._model_kwargs)
        return self._model

    async def embed(self, text: str) -> list[float]:
        import asyncio

        model = await asyncio.to_thread(self._ensure_model)
        vector = await asyncio.to_thread(model.encode, text)
        return [float(x) for x in vector]

    async def close(self) -> None:
        self._model = None


class GeminiAsyncEmbedder:
    """Embed queries with Google Gemini (``google-generativeai`` SDK).

    Mirrors the indexer's ``gemini`` xpack embedder (same default model).
    """

    _DEFAULT_MODEL = "models/embedding-001"

    def __init__(self, config) -> None:
        self._model = config.model or self._DEFAULT_MODEL
        self._api_key = config.api_key
        self._configured = False

    def _ensure_configured(self):
        import google.generativeai as genai

        if not self._configured:
            genai.configure(api_key=self._api_key)
            self._configured = True
        return genai

    async def embed(self, text: str) -> list[float]:
        import asyncio

        genai = self._ensure_configured()
        response = await asyncio.to_thread(
            genai.embed_content, model=self._model, content=text
        )
        return [float(x) for x in response["embedding"]]

    async def close(self) -> None:
        return None


class BedrockAsyncEmbedder:
    """Embed queries with AWS Bedrock (Titan models by default).

    Mirrors the indexer's ``bedrock`` xpack embedder. Credentials resolve via
    the standard AWS chain; ``region_name`` / ``aws_*`` keys on the config are
    forwarded to ``boto3``. The blocking SDK call runs in a worker thread.
    """

    _DEFAULT_MODEL = "amazon.titan-embed-text-v2:0"

    def __init__(self, config) -> None:
        extra = config.model_dump(exclude={"type", "model", "api_key"})
        self._model_id = extra.pop("model_id", None) or config.model or self._DEFAULT_MODEL
        self._client_kwargs = {k: v for k, v in extra.items() if v is not None}
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", **self._client_kwargs)
        return self._client

    async def embed(self, text: str) -> list[float]:
        import asyncio
        import json

        def invoke() -> list[float]:
            client = self._ensure_client()
            response = client.invoke_model(
                modelId=self._model_id, body=json.dumps({"inputText": text})
            )
            payload = json.loads(response["body"].read())
            return [float(x) for x in payload["embedding"]]

        return await asyncio.to_thread(invoke)

    async def close(self) -> None:
        self._client = None


# Embedder families that map onto an OpenAI-compatible async client.
_OPENAI_COMPATIBLE = {"openai", "litellm"}

_SUPPORTED = sorted(
    _OPENAI_COMPATIBLE | {"sentence_transformer", "gemini", "bedrock", "mock"}
)


def build_embedder(config) -> AsyncEmbedder:
    """Construct an :class:`AsyncEmbedder` from an ``EmbedderConfig``.

    Every embedder family supported by the indexer (via
    ``pathway.xpacks.llm.embedders``) has a matching async client here, so one
    ``embedder`` config section serves both sides.
    """

    if config.type == "mock":
        return MockAsyncEmbedder()
    if config.type in _OPENAI_COMPATIBLE:
        return OpenAIAsyncEmbedder(config)
    if config.type in {"sentence_transformer", "sentencetransformer"}:
        return SentenceTransformerAsyncEmbedder(config)
    if config.type == "gemini":
        return GeminiAsyncEmbedder(config)
    if config.type == "bedrock":
        return BedrockAsyncEmbedder(config)
    raise ValueError(
        f"Server-side embedding for type {config.type!r} is not implemented. "
        "Supported on the server: " + ", ".join(_SUPPORTED)
    )
