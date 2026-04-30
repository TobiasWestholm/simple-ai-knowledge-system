#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"
PID_FILE="$RUNTIME_DIR/embed-service.pid"
LOG_FILE="$ROOT_DIR/logs/embed-service.log"
EMBED_BIND_HOST="0.0.0.0"
EMBED_HEALTH_URL="http://127.0.0.1:8001/health"

UV_BIN="$("$ROOT_DIR/scripts/ensure_uv.sh")"

mkdir -p "$RUNTIME_DIR" "$ROOT_DIR/logs"

if curl -fsS "$EMBED_HEALTH_URL" >/dev/null 2>&1; then
  echo "Embedding service already running."
  exit 0
fi

if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" >/dev/null 2>&1; then
    echo "Embedding service process exists but health check failed; waiting for it..."
  else
    rm -f "$PID_FILE"
  fi
fi

if ! curl -fsS "$EMBED_HEALTH_URL" >/dev/null 2>&1; then
  echo "Starting host embedding service..."
  nohup "$UV_BIN" run uvicorn ai_ks.embedding_service:app --host "$EMBED_BIND_HOST" --port 8001 \
    >"$LOG_FILE" 2>&1 &
  echo "$!" >"$PID_FILE"
fi

attempt=0
until curl -fsS "$EMBED_HEALTH_URL" >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 60 ]; then
    echo "Embedding service did not become ready. Recent log output:" >&2
    tail -n 50 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "Embedding service ready on http://127.0.0.1:8001."
