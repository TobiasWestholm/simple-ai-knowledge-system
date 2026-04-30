from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from ai_ks.errors import DependencyUnavailableError


class EmbeddingServiceError(DependencyUnavailableError):
    def __init__(self, message: str) -> None:
        super().__init__("embedding", message)


@dataclass(frozen=True)
class EmbeddingServiceResult:
    model_id: str
    device: str
    embeddings: list[list[float]]
    diagnostics: dict[str, Any]


class RemoteBgeM3Embedder:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_with_details(texts).embeddings

    def embed_with_details(self, texts: list[str]) -> EmbeddingServiceResult:
        try:
            response = httpx.post(
                f"{self.base_url}/embed",
                json={"texts": texts},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingServiceError(
                f"Embedding service request failed against {self.base_url}: {exc}"
            ) from exc

        payload = response.json()
        embeddings = payload.get("embeddings", [])
        if not isinstance(embeddings, list):
            raise EmbeddingServiceError("Embedding service returned an invalid embeddings payload.")
        return EmbeddingServiceResult(
            model_id=str(payload.get("model_id", "")),
            device=str(payload.get("device", "")),
            embeddings=embeddings,
            diagnostics=dict(payload.get("diagnostics", {})),
        )
