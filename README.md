# AI Knowledge System

Local AI knowledge agent built with FastAPI, LangChain, Ollama, Qdrant, and
hybrid retrieval.

## Quick start

```bash
cp .env.example .env
make run
```

Open health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Open the chat UI:

```bash
open http://127.0.0.1:8000/
```

Run the agent endpoint:

```bash
curl http://127.0.0.1:8000/agent \
  -H 'Content-Type: application/json' \
  -d '{"message":"Summarize what FastAPI is"}'
```

Stream the agent answer:

```bash
curl -N http://127.0.0.1:8000/agent/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"Explain what the knowledge base says about the tumor microenvironment.","thread_id":"demo-thread"}'
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

## Ingestion

Create or update local indexes from the curated source list:

```bash
make ingest
```

The ingestion pipeline reads `data/sources.yaml`, loads supported files from
`knowledge/`, chunks them deterministically, embeds them with `BAAI/bge-m3`,
upserts vectors into Qdrant, and writes BM25 artifacts into `data/index/`.

## Apple Silicon runtime

On Apple Silicon, the supported local GPU path is:

- native host Ollama for `batiai/gemma4-e2b:q4`
- Docker for `api` and `qdrant`

Bring up the full stack with one command:

```bash
make run
```

That command:

- installs missing local runtime dependencies (`uv`, Ollama)
- starts host Ollama and the host embedding service
- pulls `batiai/gemma4-e2b:q4` if needed
- builds and starts `qdrant` and `api` in Docker
- waits for the API health endpoint

`api` and `qdrant` stay in Docker, while Ollama and the `BAAI/bge-m3`
embedding service run natively on the host so they can use the Apple GPU.

## Minimal UI

The app now includes a simple FastAPI-served chat UI at:

- `http://127.0.0.1:8000/`

The UI calls `POST /agent/stream` and shows:

- the streamed answer
- citations
- tool-call timeline
- final retrieval query
- request id
- timing diagnostics
- service health banner and request errors

It is intentionally small and inspectable.

If you want just the host services without Docker:

```bash
make host-up
```

To stop the stack:

```bash
make down
```

Rebuild and ingest:

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

The agent uses LangChain for orchestration while retrieval and tool business
logic remain plain Python modules underneath.

The agent also uses an in-memory LangGraph checkpointer. If you provide a
`thread_id`, repeated `/agent` or `/agent/stream` calls continue the same
server-side conversation thread. This memory is intentionally ephemeral and is
lost on restart.

`POST /agent/stream` is the streaming variant for the UI. It emits a text/event-stream
response with:

- `token` events for the final assistant answer text
- a terminal `response` event containing the full structured `AgentResponse`
- a terminal `done` event when the stream finishes

## Query workflow

`POST /query` is the direct hybrid-retrieval path for debugging and learning.
It reuses the same hybrid retrieval core as `rag_search`, but skips the full
agent runtime. The endpoint:

- runs dense retrieval in Qdrant with `BAAI/bge-m3` embeddings
- runs lexical retrieval through the BM25 artifact
- combines the ranked lists with weighted reciprocal rank fusion
- feeds the retrieved context into a small LangChain answer chain
- returns the grounded answer, citations, hits, and retrieval diagnostics

## Evaluation and validation

The current evaluation layer runs through the containerized API:

- `tool_behavior`: pass/fail evaluation over the agent's tool choices
- `failure`: input-robustness contract checks for empty, whitespace-only, and
  overlong requests, plus the 2000-character boundary case
- `timing`: per-span timing summaries aggregated from the same tool-behavior runs

The authored tool-behavior cases live in `data/evals/tool_behavior_cases.json`
and are designed to check acceptable behavior rather than one exact trace. The
evaluator checks rules such as:

- required tools were called
- forbidden tools were not called
- tool order was respected where required
- repeated tool calls were avoided unless explicitly allowed
- rewrite cases actually changed the query

The API also rejects empty, whitespace-only, and overlong queries before the
agent or retrieval stack runs. The default limit is 2000 characters and is
configured through `MAX_QUERY_CHARS`.

Future evaluation plans:

- `retrieval_hit_at_k`: assess whether hybrid retrieval surfaces at least one gold-relevant chunk or document in the top-k results, using per-query gold source labels and scoring with hit@k.
- `citation_support`: assess whether the answer's cited evidence supports its main claims, using binary `supported` / `not_supported` labels and scoring with pass/fail citation-support checks.
- `answers_the_question`: assess whether the final answer actually addresses the user's question well enough to be useful, using binary `pass` / `fail` labels and scoring with per-example answer-quality judgments.
