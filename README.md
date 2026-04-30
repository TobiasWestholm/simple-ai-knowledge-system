# AI Knowledge System

Containerized AI agent platform with FastAPI, Ollama, Qdrant, BM25, evaluation,
and monitoring. Hybrid RAG is the first tool in the agent system.

## Quick start

```bash
cp .env.example .env
make run
```

Open health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Run the agent endpoint:

```bash
curl http://127.0.0.1:8000/agent \
  -H 'Content-Type: application/json' \
  -d '{"message":"Summarize what FastAPI is"}'
```

Run the direct retrieval endpoint:

```bash
curl http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is FastAPI?"}'
```

Run the evaluation suites:

```bash
make eval
curl http://127.0.0.1:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"suites":["tool_behavior","failure","timing"]}'
```

`make eval` now uses the Dockerized API's `/evaluate` route, so evaluation runs
in the same environment as `/agent` instead of in a separate host-side CLI
process.

## Milestone 2 workflow

Create or update local indexes from the curated source list:

```bash
make ingest
```

The ingestion pipeline reads `data/sources.yaml`, loads supported files from the
`knowledge/` folder, chunks them deterministically, embeds them with `BAAI/bge-m3`,
upserts vectors into Qdrant, and writes BM25-ready artifacts into `data/index/`.

The knowledge base is intentionally simple: `.docx`, Markdown, and text files in
`knowledge/`.

The local BGE-M3 cache is intentionally stored under `.model_cache/`. The
download is large because it includes the base embedding weights; the ingestion
pipeline now limits that cache to the files needed for `sentence-transformers`
and skips the much larger ONNX export.

## Apple Silicon runtime

On this Mac, the supported local GPU path is:

- native host Ollama for `batiai/gemma4-e2b:q4`
- Docker for `api` and `qdrant`

Bring up the full stack with one command:

```bash
make run
```

That command now does all of the following:

- installs Ollama on the host if it is missing
- starts host Ollama and waits for its API
- installs `uv` on the host if it is missing
- installs and syncs the local Python dependencies with `uv`
- pulls `batiai/gemma4-e2b:q4` if it is not already available
- starts the host-native `BAAI/bge-m3` embedding service
- builds and starts `qdrant` and `api` in Docker
- waits for the API health endpoint

The app container now talks to host Ollama through
`http://host.docker.internal:11434`, which lets Ollama run on the Apple GPU
instead of inside Docker on CPU.

The host-native `BAAI/bge-m3` embedding service listens on
`http://127.0.0.1:8001` and loads the model with `device="mps"` on this Mac.
Its internal interface is:

- `GET /health`
- `POST /embed`

The containerized app now uses that host embedding service for both:

- ingestion embeddings
- query-time embeddings in hybrid retrieval

That means the `BAAI/bge-m3` model runs on the host Apple GPU for both indexing
and querying, while `qdrant` and the API remain containerized.

If you want just the host services without Docker:

```bash
make host-up
```

To stop the stack:

```bash
make down
```

Rebuild and ingest from inside the API container:

```bash
make run
make ingest
```

## Agent workflow

`POST /agent` is now the main application endpoint. It runs a small tool-calling
agent over three tools:

- `rag_search` for hybrid retrieval over the indexed knowledge base
- `rewrite_query` for retrieval-friendly query rewrites
- `summarize_context` for concise summaries

The orchestration layer is now LangChain-first: LangChain owns tool registration,
tool-calling flow, and intermediate message handling, while retrieval and tool
business logic stay in plain Python modules underneath.

## Query workflow

`POST /query` is the direct hybrid-retrieval path for debugging and learning.
It reuses the same hybrid retrieval core as `rag_search`, but skips the full
agent runtime. The endpoint:

- runs dense retrieval in Qdrant with `BAAI/bge-m3` embeddings
- runs lexical retrieval through the BM25 artifact
- combines the ranked lists with weighted reciprocal rank fusion
- feeds the retrieved context into a small LangChain answer chain
- returns the grounded answer, citations, hits, and retrieval diagnostics

This gives the project two clean layers:

- `/query` for inspecting the retrieval system directly
- `/agent` for the full LangChain tool-calling agent on top of that system

## Evaluation and validation

The current evaluation layer is intentionally narrow and runs through the
containerized API:

- `tool_behavior`: pass/fail evaluation over the agent's tool choices
- `failure`: input-robustness contract checks for empty, whitespace-only, and
  overlong requests, plus the 2000-character boundary case
- `timing`: per-span timing summaries aggregated from the same tool-behavior runs

The authored tool-behavior cases live in:

- `data/evals/tool_behavior_cases.json`

Those cases are designed to check acceptable behavior rather than one exact trace.
The evaluator checks rules such as:

- required tools were called
- forbidden tools were not called
- tool order was respected where required
- repeated tool calls were avoided unless explicitly allowed
- rewrite cases actually changed the query

The API also rejects empty, whitespace-only, and overlong queries before the
agent or retrieval stack runs. The default limit is 2000 characters and is
configured through `MAX_QUERY_CHARS`.
