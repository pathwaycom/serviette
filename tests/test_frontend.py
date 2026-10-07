"""Tests for the web chat frontend (page serving + API proxying)."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from serviette.config.schema import FrontendConfig, ServietteConfig
from serviette.frontend.main import create_app


def _mock_upstream(handler) -> httpx.AsyncClient:
    """An httpx client whose requests are served by ``handler`` (no network)."""

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://api.test"
    )


def _config(**kw) -> ServietteConfig:
    return ServietteConfig(frontend=FrontendConfig(title="MyBot", **kw))


def test_serves_chat_page():
    with TestClient(create_app(_config())) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # Title is injected into the page.
    assert "MyBot" in resp.text
    assert "__APP_TITLE__" not in resp.text


def test_config_endpoint_reports_api_url():
    with TestClient(create_app(_config(api_url="http://server:8000"))) as client:
        body = client.get("/api/v1/config").json()
    assert body == {"title": "MyBot", "api_url": "http://server:8000"}


def test_proxies_rag_to_upstream():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["url"] = str(request.url)
        captured["body"] = _json.loads(request.content)
        return httpx.Response(
            200, json={"answer": "42", "sources": [{"text": "ctx", "metadata": {}, "score": 0.9}]}
        )

    app = create_app(_config(), client=_mock_upstream(handler))
    with TestClient(app) as client:
        resp = client.post("/api/v1/rag", json={"query": "meaning?", "k": 3})

    assert resp.status_code == 200
    assert resp.json()["answer"] == "42"
    # The proxy targets the versioned API surface on the upstream server.
    assert captured["url"].endswith("/api/v1/rag")
    assert captured["body"] == {"query": "meaning?", "k": 3}


def test_proxy_forwards_upstream_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(501, json={"detail": "The /rag endpoint requires an 'llm' config section."})

    app = create_app(_config(), client=_mock_upstream(handler))
    with TestClient(app) as client:
        resp = client.post("/api/v1/rag", json={"query": "x", "k": 1})

    assert resp.status_code == 501
    assert "llm" in resp.json()["detail"]


def test_proxy_handles_unreachable_api():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    app = create_app(_config(api_url="http://down:9999"), client=_mock_upstream(handler))
    with TestClient(app) as client:
        resp = client.post("/api/v1/rag", json={"query": "x", "k": 1})

    assert resp.status_code == 502
    assert "Could not reach the API" in resp.json()["detail"]


def test_requires_frontend_section():
    with pytest.raises(ValueError, match="frontend"):
        create_app(ServietteConfig())


def test_proxies_document_search_with_its_query_string():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200, json={"total": 3, "matched": 1, "truncated": False, "documents": []}
        )

    app = create_app(_config(), client=_mock_upstream(handler))
    with TestClient(app) as client:
        resp = client.get("/api/v1/documents", params={"q": "Law", "case_sensitive": "true"})

    assert resp.status_code == 200 and resp.json()["matched"] == 1
    assert "/api/v1/documents?" in captured["url"]
    assert "q=Law" in captured["url"] and "case_sensitive=true" in captured["url"]
