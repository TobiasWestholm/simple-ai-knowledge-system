from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_ks.config import Settings, get_settings
from ai_ks.ingestion import BgeM3Embedder


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class EmbedResponse(BaseModel):
    model_id: str
    device: str
    embeddings: list[list[float]]
    diagnostics: dict[str, Any]


class EmbeddingHealthResponse(BaseModel):
    status: str
    model_id: str
    device: str


class NativeEmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = BgeM3Embedder(
            settings.embed_model_id,
            cache_dir=settings.model_cache_dir,
            device=settings.embedding_device,
        )

    def health(self) -> EmbeddingHealthResponse:
        return EmbeddingHealthResponse(
            status="alive",
            model_id=self.settings.embed_model_id,
            device=self.embedder.resolved_device(),
        )

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        started = perf_counter()
        embeddings = self.embedder.embed_texts(request.texts)
        duration_ms = round((perf_counter() - started) * 1000, 3)
        return EmbedResponse(
            model_id=self.settings.embed_model_id,
            device=self.embedder.resolved_device(),
            embeddings=embeddings,
            diagnostics={
                "text_count": len(request.texts),
                "timings_ms": {
                    "embed_texts": duration_ms,
                },
            },
        )


@lru_cache(maxsize=1)
def get_embedding_service() -> NativeEmbeddingService:
    return NativeEmbeddingService(settings=get_settings())


app = FastAPI(title="AI Knowledge System Embedding Service")


@app.get("/health", response_model=EmbeddingHealthResponse)
def health(
    service: Annotated[NativeEmbeddingService, Depends(get_embedding_service)],
) -> EmbeddingHealthResponse:
    try:
        return service.health()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/embed", response_model=EmbedResponse)
def embed(
    request: EmbedRequest,
    service: Annotated[NativeEmbeddingService, Depends(get_embedding_service)],
) -> EmbedResponse:
    try:
        return service.embed(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
