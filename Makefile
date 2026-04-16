.PHONY: sync run test lint typecheck

sync:
	uv sync --extra dev

run:
	uv run uvicorn ai_ks.main:app --reload

test:
	uv run pytest -q

lint:
	uv run ruff check .

typecheck:
	uv run mypy src
