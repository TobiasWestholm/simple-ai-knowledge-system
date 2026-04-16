from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    app_name: str = "ai-knowledge-system"

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4.1-mini"
    openai_judge_model: str = "gpt-4.1-mini"

    model_profile: str = "openai"
    embedding_profile: str = "embed_local_bge_m3"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_chunks"

    sqlite_path: Path = Field(default=Path("logs/telemetry.db"))
    log_jsonl_path: Path = Field(default=Path("logs/requests.jsonl"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
