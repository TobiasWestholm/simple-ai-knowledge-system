from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, HTTPException

from ai_ks.agent import AgentRequest, AgentResponse, LangChainAgentService, get_agent_service
from ai_ks.config import get_settings
from ai_ks.errors import DependencyUnavailableError
from ai_ks.evaluation import (
    EvaluateRequest,
    EvaluateResponse,
    EvaluationService,
    get_evaluation_service,
)
from ai_ks.ingestion import build_qdrant_client
from ai_ks.query import QueryRequest, QueryResponse, QueryService, get_query_service

settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "alive",
        "environment": settings.app_env,
        "models": {
            "llm_model_id": settings.llm_model_id,
            "llm_runtime_model": settings.llm_runtime_model,
            "llm_quantized": settings.llm_quantized,
            "embedding_model_id": settings.embed_model_id,
        },
        "services": {
            "qdrant": _qdrant_status(),
            "llm": _llm_status(),
            "embedding": _embedding_status(),
        },
        "observability": {
            "langsmith": "configured"
            if settings.langsmith_tracing and settings.langsmith_api_key
            else "disabled",
        },
    }


@app.post("/agent", response_model=AgentResponse)
def run_agent(
    request: AgentRequest,
    agent_service: Annotated[LangChainAgentService, Depends(get_agent_service)],
) -> AgentResponse:
    try:
        return agent_service.run(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DependencyUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def run_query(
    request: QueryRequest,
    query_service: Annotated[QueryService, Depends(get_query_service)],
) -> QueryResponse:
    try:
        return query_service.run(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DependencyUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/evaluate", response_model=EvaluateResponse)
def run_evaluate(
    request: EvaluateRequest,
    evaluation_service: Annotated[EvaluationService, Depends(get_evaluation_service)],
) -> EvaluateResponse:
    try:
        return evaluation_service.evaluate(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DependencyUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _qdrant_status() -> str:
    try:
        client = build_qdrant_client(settings.qdrant_url)
        client.get_collections()
        return "reachable"
    except Exception:
        return "unreachable"


def _llm_status() -> str:
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


def _embedding_status() -> str:
    try:
        response = httpx.get(
            f"{settings.embedding_url.rstrip('/')}/health",
            timeout=1.5,
        )
        response.raise_for_status()
        return "available"
    except Exception:
        return "unreachable"
