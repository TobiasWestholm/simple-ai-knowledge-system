FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
RUN uv sync --frozen || uv sync

COPY src ./src
COPY data ./data

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "ai_ks.main:app", "--host", "0.0.0.0", "--port", "8000"]
