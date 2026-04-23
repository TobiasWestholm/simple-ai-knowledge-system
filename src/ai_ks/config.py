from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    app_name: str = "ai-knowledge-system"

    llm_backend: Literal["ollama", "transformers"] = "ollama"
    llm_model_id: str = "google/gemma-4-E4B-it"
    llm_runtime_model: str = "gemma4:e4b"
    llm_quantized: bool = True
    ollama_url: str = "http://localhost:11434"

    embed_model_id: str = "BAAI/bge-m3"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_chunks"

    data_dir: Path = Field(default=Path("data"))
    index_dir: Path = Field(default=Path("data/index"))
    sources_path: Path = Field(default=Path("data/sources.yaml"))
    model_cache_dir: Path = Field(default=Path(".model_cache"))
    chunk_size: int = 900
    chunk_overlap: int = 150

    sqlite_path: Path = Field(default=Path("logs/telemetry.db"))
    log_jsonl_path: Path = Field(default=Path("logs/requests.jsonl"))
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
