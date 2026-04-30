#!/bin/sh
set -eu

UV_BIN_CANDIDATE="$HOME/.local/bin/uv"

find_uv_bin() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi
  if [ -x "$UV_BIN_CANDIDATE" ]; then
    printf '%s\n' "$UV_BIN_CANDIDATE"
    return 0
  fi
  return 1
}

if ! UV_BIN="$(find_uv_bin)"; then
  echo "Installing uv with the official installer..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="$(find_uv_bin)"
fi

printf '%s\n' "$UV_BIN"
