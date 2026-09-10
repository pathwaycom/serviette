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
        from serviette.testing import fake_embedding

        return fake_embedding(text)

    async def close(self) -> None:
        return None


# Config keys that are consumed by the schema / the indexer's UDF wrapper and
# must never reach a provider SDK on the server side. ``batch_size``,
# ``truncation_keep_strategy``, ``capacity`` and ``retries`` are indexer
# throughput knobs (pathway UDF executor / xpack constructor arguments).
_SCHEMA_KEYS = {"type", "model", "api_key", "query_prefix", "document_prefix"}
_INDEXER_ONLY_KEYS = {"batch_size", "truncation_keep_strategy", "capacity", "retries"}


def _extra_kwargs(config) -> dict:
    """Extra config keys, minus schema fields and indexer-only knobs.

    Where each remaining key goes (client constructor vs. per-call) is decided
    per family below — the split must mirror what the indexer's xpack embedder
    does with the *same* section, or query and document vectors diverge.
    """
    extra = config.model_dump(exclude=_SCHEMA_KEYS | _INDEXER_ONLY_KEYS)
    return {k: v for k, v in extra.items() if v is not None}


class OpenAIAsyncEmbedder:
    """Embed queries with OpenAI (or any OpenAI-compatible endpoint).

    Mirrors the indexer's ``OpenAIEmbedder`` xpack, which forwards extra keys
    to ``embeddings.create`` per call (``dimensions``, ``encoding_format``,
    ``user``). The handful of keys that belong to the client object
    (``base_url``, ``organization``, ...) are routed there instead.
    """

    _CLIENT_KEYS = frozenset({
        "base_url",
        "organization",
        "project",
        "timeout",
        "max_retries",
        "default_headers",
        "default_query",
    })

    def __init__(self, config) -> None:
        self._model = config.model or "text-embedding-3-small"
        self._api_key = config.api_key
        extra = _extra_kwargs(config)
        self._client_kwargs = {k: v for k, v in extra.items() if k in self._CLIENT_KEYS}
        self._call_kwargs = {k: v for k, v in extra.items() if k not in self._CLIENT_KEYS}
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key, **self._client_kwargs)
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._ensure_client()
        resp = await client.embeddings.create(
            model=self._model, input=[text], **self._call_kwargs
        )
        return list(resp.data[0].embedding)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


class LiteLLMAsyncEmbedder:
    """Embed queries through LiteLLM (``type: litellm``).

    Unlike :class:`OpenAIAsyncEmbedder` this calls ``litellm.aembedding``, so
    the provider prefix in ``model`` (``openrouter/...``, ``cohere/...``, …)
    selects the endpoint and ``api_key`` is forwarded to *that* provider. This
    mirrors the indexer, which embeds documents with pathway's
    ``LiteLLMEmbedder`` for the same config section — routing the server's
    query embeddings through the plain OpenAI client instead would send the
    key to api.openai.com (401) or, with an OpenAI key, silently embed queries
    with a different model than the documents.
    """

    def __init__(self, config) -> None:
        if not config.model:
            raise ValueError(
                "embedder type 'litellm' requires a 'model' with a provider prefix, "
                "e.g. 'openrouter/qwen/qwen3-embedding-8b' (LiteLLM has no default)."
            )
        self._model = config.model
        self._api_key = config.api_key
        # Extra keys are per-call kwargs for litellm.aembedding (api_base,
        # dimensions, ...) — there is no persistent client object to construct.
        self._call_kwargs = _extra_kwargs(config)

    async def embed(self, text: str) -> list[float]:
        import litellm

        resp = await litellm.aembedding(
            model=self._model,
            api_key=self._api_key,
            input=[text],
            **self._call_kwargs,
        )
        return [float(x) for x in resp.data[0]["embedding"]]

    async def close(self) -> None:
        # No persistent client; nothing to release.
        return None


class SentenceTransformerAsyncEmbedder:
    """Embed queries with a local sentence-transformers model.

    ``is_local = True`` opts into the server's startup warm-up: the first
    real user question must not pay the torch import + model load.

    Fully local — no provider, no credentials. The (CPU/GPU-bound) encode runs
    off the event loop in a worker thread. Mirrors the indexer's
    ``sentence_transformer`` xpack embedder; use the same ``model`` on both
    sides.
    """

    _DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    is_local = True

    def __init__(self, config) -> None:
        self._model_name = config.model or self._DEFAULT_MODEL
        # Forward extra config keys (device, truncate_dim, ...) to the
        # SentenceTransformer constructor — mirrors the indexer's xpack
        # embedder, so e.g. Matryoshka truncation stays consistent.
        # Same split as the xpack: ``call_kwargs`` are per-``encode`` options
        # (normalize_embeddings, ...), ``batch_size`` is indexer-only, the rest
        # are SentenceTransformer constructor kwargs.
        self._model_kwargs = _extra_kwargs(config)
        self._call_kwargs = dict(self._model_kwargs.pop("call_kwargs", None) or {})
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
        vector = await asyncio.to_thread(model.encode, text, **self._call_kwargs)
        return [float(x) for x in vector]

    async def close(self) -> None:
        self._model = None


class GeminiAsyncEmbedder:
    """Embed queries with Google Gemini (``google-generativeai`` SDK).

    Mirrors the indexer's ``gemini`` xpack embedder: same default model, and
    extra keys (``task_type``, ``output_dimensionality``, ...) go to
    ``embed_content`` per call exactly as there.
    """

    _DEFAULT_MODEL = "models/embedding-001"

    def __init__(self, config) -> None:
        self._model = config.model or self._DEFAULT_MODEL
        self._api_key = config.api_key
        self._call_kwargs = _extra_kwargs(config)
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
            genai.embed_content, model=self._model, content=text, **self._call_kwargs
        )
        return [float(x) for x in response["embedding"]]

    async def close(self) -> None:
        return None


class BedrockAsyncEmbedder:
    """Embed queries with AWS Bedrock (Titan models by default).

    Mirrors the indexer's ``bedrock`` xpack embedder: ``model`` (or the
    xpack's ``model_id``) picks the model, credentials/region keys go to the
    boto3 client, and the model-specific request options (Titan:
    ``dimensions``/``normalize``; Cohere: ``input_type``/``truncate``) go into
    the request body — the same shapes the indexer sends. Credentials resolve
    via the standard AWS chain when omitted. The blocking SDK call runs in a
    worker thread.
    """

    _DEFAULT_MODEL = "amazon.titan-embed-text-v2:0"
    _SESSION_KEYS = frozenset({
        "region_name",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "endpoint_url",
    })

    def __init__(self, config) -> None:
        extra = _extra_kwargs(config)
        self._model_id = extra.pop("model_id", None) or config.model or self._DEFAULT_MODEL
        self._client_kwargs = {k: v for k, v in extra.items() if k in self._SESSION_KEYS}
        self._request_kwargs = {k: v for k, v in extra.items() if k not in self._SESSION_KEYS}
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", **self._client_kwargs)
        return self._client

    def _request_body(self, text: str) -> dict:
        model = self._model_id.lower()
        opts = self._request_kwargs
        if "cohere" in model:
            body: dict = {
                "texts": [text],
                # Queries, not documents: the indexer's default is search_document.
                "input_type": opts.get("input_type", "search_query"),
            }
            if "truncate" in opts:
                body["truncate"] = opts["truncate"]
            return body
        body = {"inputText": text}
        if "titan" in model:
            for key in ("dimensions", "normalize"):
                if key in opts:
                    body[key] = opts[key]
        return body

    def _parse_embedding(self, payload: dict) -> list[float]:
        if "cohere" in self._model_id.lower():
            embeddings = payload.get("embeddings") or [[]]
            return [float(x) for x in embeddings[0]]
        return [float(x) for x in payload["embedding"]]

    async def embed(self, text: str) -> list[float]:
        import asyncio
        import json

        def invoke() -> list[float]:
            client = self._ensure_client()
            response = client.invoke_model(
                modelId=self._model_id,
                body=json.dumps(self._request_body(text)),
                contentType="application/json",
                accept="application/json",
            )
            return self._parse_embedding(json.loads(response["body"].read()))

        return await asyncio.to_thread(invoke)

    async def close(self) -> None:
        self._client = None


_SUPPORTED = sorted(
    {"openai", "litellm", "sentence_transformer", "gemini", "bedrock", "mock"}
)


def build_embedder(config) -> AsyncEmbedder:
    """Construct an :class:`AsyncEmbedder` from an ``EmbedderConfig``.

    Every embedder family supported by the indexer (via
    ``pathway.xpacks.llm.embedders``) has a matching async client here, so one
    ``embedder`` config section serves both sides.
    """

    if config.type == "mock":
        return MockAsyncEmbedder()
    if config.type == "litellm":
        return LiteLLMAsyncEmbedder(config)
    if config.type == "openai":
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
