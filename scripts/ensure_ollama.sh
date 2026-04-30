#!/bin/sh
set -eu

OLLAMA_APP="/Applications/Ollama.app"
OLLAMA_APP_BIN="$OLLAMA_APP/Contents/Resources/ollama"
OLLAMA_LINK_BIN="/usr/local/bin/ollama"

find_ollama_bin() {
  if command -v ollama >/dev/null 2>&1; then
    command -v ollama
    return 0
  fi
  if [ -x "$OLLAMA_LINK_BIN" ]; then
    printf '%s\n' "$OLLAMA_LINK_BIN"
    return 0
  fi
  if [ -x "$OLLAMA_APP_BIN" ]; then
    printf '%s\n' "$OLLAMA_APP_BIN"
    return 0
  fi
  return 1
}

if ! OLLAMA_BIN="$(find_ollama_bin)"; then
  echo "Installing Ollama with the official macOS installer..." >&2
  curl -fsSL https://ollama.com/install.sh | sh
  OLLAMA_BIN="$(find_ollama_bin)"
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama on the host..." >&2
  open -a Ollama --args hidden
fi

attempt=0
until curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 30 ]; then
    echo "Ollama did not become ready on http://127.0.0.1:11434." >&2
    exit 1
  fi
  sleep 1
done

printf '%s\n' "$OLLAMA_BIN"
