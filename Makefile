.PHONY: sync run up down host-up test lint typecheck compose-up model-bootstrap embed-service ingest health eval

sync:
	"$$(./scripts/ensure_uv.sh)" sync --extra dev

run: up

up:
	./scripts/up_local_stack.sh

down:
	docker compose down
	./scripts/stop_embedding_service.sh

host-up:
	./scripts/ensure_ollama.sh
	./scripts/start_embedding_service.sh

test:
	"$$(./scripts/ensure_uv.sh)" run pytest -q

lint:
	"$$(./scripts/ensure_uv.sh)" run ruff check .

typecheck:
	"$$(./scripts/ensure_uv.sh)" run mypy src

compose-up:
	docker compose up -d --build qdrant api

model-bootstrap:
	./scripts/ensure_ollama.sh >/dev/null
	"$$(./scripts/ensure_ollama.sh)" pull batiai/gemma4-e2b:q4

embed-service:
	./scripts/start_embedding_service.sh

ingest:
	docker compose run --rm api ai-ks ingest

health:
	curl http://127.0.0.1:8000/health

eval:
	curl -sS http://127.0.0.1:8000/evaluate \
		-H 'Content-Type: application/json' \
		-d '{"suites":["tool_behavior","failure","timing"]}'
