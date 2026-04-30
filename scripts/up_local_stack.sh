#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="$("$ROOT_DIR/scripts/ensure_uv.sh")"
OLLAMA_BIN="$("$ROOT_DIR/scripts/ensure_ollama.sh")"

echo "Syncing local Python dependencies..."
"$UV_BIN" sync --extra dev

if ! "$OLLAMA_BIN" list | grep -q '^batiai/gemma4-e2b:q4[[:space:]]'; then
  echo "Pulling batiai/gemma4-e2b:q4..."
  "$OLLAMA_BIN" pull batiai/gemma4-e2b:q4
else
  echo "Ollama model batiai/gemma4-e2b:q4 already present."
fi

"$ROOT_DIR/scripts/start_embedding_service.sh"

echo "Starting Docker services..."
docker compose up -d --build qdrant api

echo "Waiting for API health endpoint..."
attempt=0
until curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 60 ]; then
    echo "API did not become ready on http://127.0.0.1:8000/health." >&2
    exit 1
  fi
  sleep 1
done

echo "Full stack is ready."
