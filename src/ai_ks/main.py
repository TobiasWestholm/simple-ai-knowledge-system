from typing import Any

import httpx
from fastapi import FastAPI

from ai_ks.config import get_settings
from ai_ks.ingestion import build_qdrant_client

settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "alive",
        "environment": settings.app_env,
        "models": {
            "llm_backend": settings.llm_backend,
            "llm_model_id": settings.llm_model_id,
            "llm_runtime_model": settings.llm_runtime_model,
            "llm_quantized": settings.llm_quantized,
            "embedding_model_id": settings.embed_model_id,
        },
        "services": {
            "qdrant": _qdrant_status(),
            "llm": _llm_status(),
        },
        "observability": {
            "langfuse": "configured" if settings.langfuse_host else "disabled",
        },
    }


def _qdrant_status() -> str:
    try:
        client = build_qdrant_client(settings.qdrant_url)
        client.get_collections()
        return "reachable"
    except Exception:
        return "unreachable"


def _llm_status() -> str:
    if settings.llm_backend != "ollama":
        return "configured"

    try:
        response = httpx.get(
            f"{settings.ollama_url.rstrip('/')}/api/tags",
            timeout=1.5,
        )
        response.raise_for_status()
        payload = response.json()
        models = {
            model.get("name", "")
            for model in payload.get("models", [])
            if isinstance(model, dict)
        }
        if settings.llm_runtime_model in models:
            return "available"
        return "reachable"
    except Exception:
        return "unreachable"
