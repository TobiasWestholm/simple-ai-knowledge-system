# AI Knowledge System

Portfolio-grade hybrid RAG service with FastAPI, Qdrant, BM25, evaluation, and monitoring.

## Quick start

```bash
uv sync --extra dev
uv run uvicorn ai_ks.main:app --reload
```

Open health endpoint:

```bash
curl http://127.0.0.1:8000/health
```
