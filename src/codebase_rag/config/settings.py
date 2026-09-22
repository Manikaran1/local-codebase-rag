"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_dir: Path = Path(".codebase-rag")
    log_level: str = "INFO"

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = ""

    embedding_provider: str = "sentence-transformers"
    embedding_model: str = ""

    model_config = SettingsConfigDict(
        env_prefix="CODEBASE_RAG_",
        env_file=".env",
        extra="ignore",
    )
