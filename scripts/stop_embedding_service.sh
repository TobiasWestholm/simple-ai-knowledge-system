#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PID_FILE="$ROOT_DIR/.runtime/embed-service.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No embedding service pid file found."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" >/dev/null 2>&1; then
  kill "$PID"
  echo "Stopped embedding service (pid $PID)."
else
  echo "Embedding service pid $PID was not running."
fi

rm -f "$PID_FILE"
