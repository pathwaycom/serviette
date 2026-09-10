"""One ``embedder`` section, two consumers: the indexer (pathway xpack UDF) and
the server (async SDK client) must agree on where every config key goes.

Regression suite for the family of bugs behind issue #2: a key the indexer
accepts (``dimensions``, ``call_kwargs``, ``model`` for bedrock, ...) must not
crash the server on the first query or, worse, be silently dropped so that
queries and documents are embedded with different models. Every test runs
offline against fakes of the provider SDKs; the indexer-side checks import
pathway but never run a graph.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from serviette.config.schema import EmbedderConfig
from serviette.quickstart.wizard import ScriptedPrompter, Wizard
from serviette.server.embedder import build_embedder as build_server_embedder

INDEXER_ONLY_KEYS = ("batch_size", "truncation_keep_strategy", "retries", "capacity")


def _indexer_embedder(cfg: EmbedderConfig):
    pytest.importorskip("pathway")
    from serviette.indexer.graph import build_xpack_embedder

    return build_xpack_embedder(cfg)


# ---------------------------------------------------------------------------
# openai
# ---------------------------------------------------------------------------


async def test_openai_server_splits_client_and_per_call_kwargs(monkeypatch):
    """The indexer sends extras to ``embeddings.create``; the server must too.

    ``dimensions`` is a per-call parameter, ``base_url`` a client one; indexer
    tuning knobs (batch size, retries, ...) are dropped rather than crashing
    ``AsyncOpenAI.__init__``.
    """
    import openai

    created: dict = {}
    calls: dict = {}

    class FakeEmbeddings:
        async def create(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    class FakeClient:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.embeddings = FakeEmbeddings()

        async def close(self):
            return None

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)

    embedder = build_server_embedder(
        EmbedderConfig(
            type="openai",
            model="text-embedding-3-large",
            api_key="sk-test",
            dimensions=256,
            base_url="http://localhost:11434/v1",
            batch_size=64,
            truncation_keep_strategy="end",
            retries=3,
            capacity=8,
        )
    )
    assert await embedder.embed("q") == [0.1, 0.2]

    assert created["api_key"] == "sk-test"
    assert created["base_url"] == "http://localhost:11434/v1"
    assert calls["model"] == "text-embedding-3-large"
    assert calls["input"] == ["q"]
    assert calls["dimensions"] == 256
    assert "dimensions" not in created and "base_url" not in calls
    for key in INDEXER_ONLY_KEYS:
        assert key not in created and key not in calls


def test_openai_indexer_rejects_base_url_with_guidance():
    """pathway's OpenAIEmbedder cannot take ``base_url``: it would go to
    ``embeddings.create`` and fail on the first chunk. Fail at build time and
    say what to do instead."""
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        _indexer_embedder(
            EmbedderConfig(type="openai", api_key="sk-test", base_url="http://localhost:11434/v1")
        )


# ---------------------------------------------------------------------------
# litellm
# ---------------------------------------------------------------------------


async def test_litellm_both_sides_forward_extras_per_call(monkeypatch):
    import litellm

    captured: dict = {}

    async def fake_aembedding(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(data=[{"embedding": [0.5]}])

    monkeypatch.setattr(litellm, "aembedding", fake_aembedding)

    cfg = EmbedderConfig(
        type="litellm",
        model="openai/text-embedding-3-small",
        api_key="sk-test",
        api_base="http://localhost:11434/v1",
        dimensions=256,
        retries=2,
    )
    server = build_server_embedder(cfg)
    assert await server.embed("q") == [0.5]
    assert captured["api_base"] == "http://localhost:11434/v1"
    assert captured["dimensions"] == 256
    assert "retries" not in captured

    udf = _indexer_embedder(cfg)
    assert udf.kwargs["model"] == "openai/text-embedding-3-small"
    assert udf.kwargs["api_base"] == "http://localhost:11434/v1"
    assert udf.kwargs["dimensions"] == 256


def test_litellm_indexer_requires_model():
    """Same fail-fast as the server: LiteLLM has no default embedding model."""
    with pytest.raises(ValueError, match="model"):
        _indexer_embedder(EmbedderConfig(type="litellm", api_key="sk-test"))


# ---------------------------------------------------------------------------
# bedrock
# ---------------------------------------------------------------------------


def test_bedrock_indexer_honours_model():
    """``model`` (the schema's field) must reach BedrockEmbedder as ``model_id``;
    previously it landed in the ignored kwargs and Titan v2 was used regardless."""
    udf = _indexer_embedder(
        EmbedderConfig(type="bedrock", model="cohere.embed-english-v3", region_name="us-east-1")
    )
    assert udf.kwargs["model_id"] == "cohere.embed-english-v3"
    assert "model" not in udf.kwargs

    # The xpack's own spelling keeps working too.
    udf = _indexer_embedder(EmbedderConfig(type="bedrock", model_id="amazon.titan-embed-text-v1"))
    assert udf.kwargs["model_id"] == "amazon.titan-embed-text-v1"


def _fake_boto3(monkeypatch, response_body: dict):
    import boto3

    created: dict = {}
    invoked: dict = {}

    class FakeClient:
        def invoke_model(self, **kwargs):
            invoked.update(kwargs)
            return {"body": SimpleNamespace(read=lambda: json.dumps(response_body).encode())}

    def client(service, **kwargs):
        created["service"] = service
        created.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(boto3, "client", client)
    return created, invoked


async def test_bedrock_server_splits_session_and_request_kwargs(monkeypatch):
    """Titan: ``dimensions``/``normalize`` belong in the request body (as on the
    indexer), AWS keys to the boto3 client."""
    created, invoked = _fake_boto3(monkeypatch, {"embedding": [1.0, 2.0]})

    embedder = build_server_embedder(
        EmbedderConfig(
            type="bedrock",
            model="amazon.titan-embed-text-v2:0",
            region_name="us-east-1",
            aws_access_key_id="AKIA-test",
            dimensions=256,
            normalize=True,
        )
    )
    assert await embedder.embed("q") == [1.0, 2.0]

    assert created["service"] == "bedrock-runtime"
    assert created["region_name"] == "us-east-1"
    assert created["aws_access_key_id"] == "AKIA-test"
    assert "dimensions" not in created and "normalize" not in created
    assert invoked["modelId"] == "amazon.titan-embed-text-v2:0"
    assert json.loads(invoked["body"]) == {"inputText": "q", "dimensions": 256, "normalize": True}


async def test_bedrock_server_supports_cohere_like_indexer(monkeypatch):
    """The indexer speaks Cohere's request/response shape; the server must too,
    with the query-side ``input_type``."""
    _created, invoked = _fake_boto3(monkeypatch, {"embeddings": [[3.0, 4.0]]})

    embedder = build_server_embedder(
        EmbedderConfig(type="bedrock", model="cohere.embed-english-v3", region_name="us-east-1")
    )
    assert await embedder.embed("q") == [3.0, 4.0]
    assert json.loads(invoked["body"]) == {"texts": ["q"], "input_type": "search_query"}


# ---------------------------------------------------------------------------
# gemini
# ---------------------------------------------------------------------------


async def test_gemini_server_forwards_extras_to_embed_content(monkeypatch):
    """The indexer passes ``task_type``/``output_dimensionality`` to
    ``embed_content``; the server dropped every extra key."""
    configured: dict = {}
    calls: dict = {}

    fake = types.ModuleType("google.generativeai")
    fake.configure = lambda **kwargs: configured.update(kwargs)

    def embed_content(model, content, **kwargs):
        calls.update({"model": model, "content": content, **kwargs})
        return {"embedding": [0.25]}

    fake.embed_content = embed_content
    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = fake
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake)

    embedder = build_server_embedder(
        EmbedderConfig(
            type="gemini",
            model="models/text-embedding-004",
            api_key="g-test",
            task_type="retrieval_query",
            output_dimensionality=256,
            retries=2,
        )
    )
    assert await embedder.embed("q") == [0.25]
    assert configured == {"api_key": "g-test"}
    assert calls["model"] == "models/text-embedding-004"
    assert calls["content"] == "q"
    assert calls["task_type"] == "retrieval_query"
    assert calls["output_dimensionality"] == 256
    assert "retries" not in calls


# ---------------------------------------------------------------------------
# sentence_transformer
# ---------------------------------------------------------------------------


async def test_sentence_transformer_server_mirrors_indexer_kwargs(monkeypatch):
    """``call_kwargs`` go to ``encode`` (as in the xpack), ``batch_size`` is
    indexer-only, everything else is a constructor kwarg."""
    created: dict = {}
    encoded: dict = {}

    class FakeModel:
        def __init__(self, name, **kwargs):
            created["name"] = name
            created.update(kwargs)

        def encode(self, text, **kwargs):
            encoded["text"] = text
            encoded.update(kwargs)
            return [0.75]

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    embedder = build_server_embedder(
        EmbedderConfig(
            type="sentence_transformer",
            model="intfloat/e5-small-v2",
            device="cuda",
            truncate_dim=256,
            batch_size=32,
            call_kwargs={"normalize_embeddings": True},
        )
    )
    assert await embedder.embed("q") == [0.75]
    assert created == {"name": "intfloat/e5-small-v2", "device": "cuda", "truncate_dim": 256}
    assert encoded == {"text": "q", "normalize_embeddings": True}


# ---------------------------------------------------------------------------
# quickstart
# ---------------------------------------------------------------------------


def test_wizard_requires_model_for_litellm(tmp_path, monkeypatch):
    """LiteLLM has no default model, so the wizard must not accept a blank one
    (the generated config would fail on both sides)."""
    monkeypatch.chdir(tmp_path)
    tokens = iter(
        [
            "2",  # config type: indexer only
            "1", "/data/a",  # one filesystem source
            "6",  # done
            "1",  # vector db: duckdb
            "3",  # embedder: litellm
            "",  # model: blank -> rejected, asked again
            "openrouter/qwen/qwen3-embedding-8b",
            "",  # api key: keep the env reference
            "K",  # license key
            "",  # output path
        ]
    )
    prompter = ScriptedPrompter(input_fn=lambda _p: next(tokens), output_fn=lambda _s: None)
    answers = Wizard(prompter=prompter).run()
    assert answers["embedder_type"] == "litellm"
    assert answers["embedder_model"] == "openrouter/qwen/qwen3-embedding-8b"
    assert answers["embedder_api_key"] == "${OPENAI_API_KEY}"
