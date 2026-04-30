from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MAX_QUERY_CHARS = 2000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    app_name: str = "ai-knowledge-system"

    llm_model_id: str = "batiai/gemma4-e2b"
    llm_runtime_model: str = "batiai/gemma4-e2b:q4"
    llm_quantized: bool = True
    ollama_url: str = "http://127.0.0.1:11434"

    embed_model_id: str = "BAAI/bge-m3"
    embedding_url: str = "http://127.0.0.1:8001"
    embedding_device: str = "mps"
    embedding_timeout_seconds: float = 60.0

    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "knowledge_chunks"
    qdrant_upsert_max_payload_bytes: int = 24 * 1024 * 1024
    qdrant_upsert_max_points: int = 128

    data_dir: Path = Field(default=Path("data"))
    index_dir: Path = Field(default=Path("data/index"))
    sources_path: Path = Field(default=Path("data/sources.yaml"))
    model_cache_dir: Path = Field(default=Path(".model_cache"))
    eval_dir: Path = Field(default=Path("data/evals"))
    chunk_size: int = 900
    chunk_overlap: int = 150
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS
    retrieval_limit: int = 5
    retrieval_candidate_limit: int = 8
    retrieval_semantic_weight: float = 0.7
    retrieval_lexical_weight: float = 0.3
    retrieval_rrf_k: int = 60
    agent_max_steps: int = 4

    sqlite_path: Path = Field(default=Path("logs/telemetry.db"))
    log_jsonl_path: Path = Field(default=Path("logs/requests.jsonl"))
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "ai-knowledge-system"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
