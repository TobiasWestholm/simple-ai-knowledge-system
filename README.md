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

## Milestone 2 workflow

Create or update local indexes from the curated source list:

```bash
uv run ai-ks ingest
```

The ingestion pipeline reads `data/sources.yaml`, chunks documents deterministically,
embeds them with `BAAI/bge-m3`, upserts vectors into Qdrant, and writes BM25-ready
artifacts into `data/index/`.

`sources.yaml` supports individual files, URLs, and whole directories. Word
documents (`.docx`) are extracted directly during ingestion, which makes it easy
to build a knowledge base from a folder of study notes or reports.

If Docker is unavailable, you can point `QDRANT_URL` to a local embedded store
using `local:data/qdrant`.

The local BGE-M3 cache is intentionally stored under `.model_cache/`. The
download is large because it includes the base embedding weights; the ingestion
pipeline now limits that cache to the files needed for `sentence-transformers`
and skips the much larger ONNX export.

## Docker workflow

Bring up the backing services:

```bash
docker compose up -d qdrant api
```

Rebuild and ingest from inside the API container:

```bash
docker compose up -d --build qdrant api
docker compose run --rm api ai-ks ingest
```
