from __future__ import annotations

from typing import Any

import httpx
from fastapi.testclient import TestClient

from ai_ks.embedding_service import (
    EmbeddingHealthResponse,
    EmbedRequest,
    EmbedResponse,
    NativeEmbeddingService,
    app,
    get_embedding_service,
)
from ai_ks.embeddings import EmbeddingServiceError, RemoteBgeM3Embedder


class FakeNativeEmbeddingService:
    def health(self) -> EmbeddingHealthResponse:
        return EmbeddingHealthResponse(
            status="alive",
            model_id="BAAI/bge-m3",
            device="mps:0",
        )

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(
            model_id="BAAI/bge-m3",
            device="mps:0",
            embeddings=[[1.0, 2.0] for _ in request.texts],
            diagnostics={
                "text_count": len(request.texts),
                "timings_ms": {"embed_texts": 12.5},
            },
        )


def test_embedding_service_health_endpoint_uses_dependency_override() -> None:
    app.dependency_overrides[get_embedding_service] = lambda: FakeNativeEmbeddingService()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "alive"
    assert payload["model_id"] == "BAAI/bge-m3"
    assert payload["device"] == "mps:0"

    app.dependency_overrides.clear()


def test_embedding_service_embed_endpoint_uses_dependency_override() -> None:
    app.dependency_overrides[get_embedding_service] = lambda: FakeNativeEmbeddingService()
    client = TestClient(app)

    response = client.post("/embed", json={"texts": ["hello", "world"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "BAAI/bge-m3"
    assert payload["device"] == "mps:0"
    assert len(payload["embeddings"]) == 2
    assert payload["diagnostics"]["text_count"] == 2

    app.dependency_overrides.clear()


def test_remote_embedder_calls_host_service(monkeypatch: Any) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "model_id": "BAAI/bge-m3",
                "device": "mps:0",
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                "diagnostics": {"text_count": 2},
            }

    recorded: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> FakeResponse:
        recorded["url"] = url
        recorded["json"] = json
        recorded["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    embedder = RemoteBgeM3Embedder(
        base_url="http://host.docker.internal:8001",
        timeout_seconds=30.0,
    )
    embeddings = embedder.embed_texts(["hello", "world"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert recorded["url"] == "http://host.docker.internal:8001/embed"
    assert recorded["json"] == {"texts": ["hello", "world"]}
    assert recorded["timeout"] == 30.0


def test_remote_embedder_wraps_http_errors(monkeypatch: Any) -> None:
    def fake_post(url: str, json: dict[str, Any], timeout: float) -> Any:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)

    embedder = RemoteBgeM3Embedder(
        base_url="http://host.docker.internal:8001",
        timeout_seconds=30.0,
    )

    try:
        embedder.embed_texts(["hello"])
    except EmbeddingServiceError as exc:
        assert exc.service == "embedding"
        assert "connection refused" in str(exc)
    else:
        raise AssertionError("Expected EmbeddingServiceError for a failed HTTP call")


def test_native_embedding_service_reports_requested_device(monkeypatch: Any) -> None:
    class FakeEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0] for _ in texts]

        def resolved_device(self) -> str:
            return "mps:0"

    service = NativeEmbeddingService.__new__(NativeEmbeddingService)
    service.settings = type(
        "SettingsLike",
        (),
        {"embed_model_id": "BAAI/bge-m3"},
    )()
    service.embedder = FakeEmbedder()

    health = service.health()
    response = service.embed(EmbedRequest(texts=["hello"]))

    assert health.device == "mps:0"
    assert response.device == "mps:0"
    assert response.model_id == "BAAI/bge-m3"
