"""Application configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

DEFAULT_IGNORE_PATTERNS = (
    ".git/",
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".venv/",
    "venv/",
    "node_modules/",
    "build/",
    "dist/",
    "*.egg-info/",
    ".codebase-rag/",
    ".vscode/",
    ".idea/",
    "*.swp",
    "*.swo",
    ".DS_Store",
    "Thumbs.db",
)

DEFAULT_SUPPORTED_EXTENSIONS: Mapping[str, str] = MappingProxyType(
    {
        ".py": "Python",
        ".md": "Markdown",
        ".markdown": "Markdown",
        ".txt": "text",
        ".rst": "reStructuredText",
        ".toml": "TOML",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".json": "JSON",
        ".html": "HTML",
        ".css": "CSS",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".java": "Java",
        ".go": "Go",
        ".rs": "Rust",
        ".c": "C",
        ".h": "C",
        ".cpp": "C++",
        ".hpp": "C++",
        ".cs": "C#",
        ".sh": "Shell",
        ".sql": "SQL",
        ".xml": "XML",
    }
)


class Settings(BaseSettings):
    data_dir: Path = Path(".codebase-rag")
    log_level: str = "INFO"

    max_file_size_bytes: int = Field(default=DEFAULT_MAX_FILE_SIZE_BYTES, gt=0)
    max_chunk_chars: int = Field(default=4000, gt=0)
    default_ignore_patterns: tuple[str, ...] = DEFAULT_IGNORE_PATTERNS
    supported_extensions: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_SUPPORTED_EXTENSIONS)
    )

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = ""

    embedding_provider: str = "sentence-transformers"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_revision: str | None = None
    embedding_allow_download: bool = False
    embedding_batch_size: int = Field(default=32, gt=0)
    vector_backend: str = "sqlite"
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)

    retrieval_candidate_limit: int = Field(default=40, gt=0)
    retrieval_fusion_strategy: Literal["weighted", "rrf"] = "weighted"
    retrieval_vector_weight: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    retrieval_lexical_weight: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    retrieval_rrf_k: int = Field(default=60, gt=0)
    retrieval_max_hits_per_file: int = Field(default=3, gt=0)
    retrieval_overlap_line_threshold: float = Field(
        default=0.8, ge=0, le=1, allow_inf_nan=False
    )
    retrieval_overlap_text_threshold: float = Field(
        default=0.8, ge=0, le=1, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def validate_retrieval_weights(self) -> Settings:
        if self.retrieval_vector_weight + self.retrieval_lexical_weight <= 0:
            raise ValueError("the sum of retrieval fusion weights must be greater than zero")
        return self

    @property
    def index_db_path(self) -> Path:
        """Return the shared local index database path."""
        return self.data_dir / "indexes" / "index.sqlite3"

    model_config = SettingsConfigDict(
        env_prefix="CODEBASE_RAG_",
        env_file=".env",
        extra="ignore",
    )
