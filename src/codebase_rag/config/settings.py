"""Application configuration."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024

DEFAULT_IGNORE_PATTERNS = (
    ".git/",
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".venv/",
    "venv/",
    "env/",
    "node_modules/",
    "build/",
    "dist/",
    "*.egg-info/",
    ".codebase-rag/",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    ".vscode/",
    ".idea/",
    "*.swp",
    "*.swo",
    ".DS_Store",
    "Thumbs.db",
    "models/",
    "indexes/",
    "vector_store/",
)

DEFAULT_SUPPORTED_EXTENSIONS: Mapping[str, str] = MappingProxyType({
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
})


class Settings(BaseSettings):
    data_dir: Path = Path(".codebase-rag")
    log_level: str = "INFO"

    max_file_size_bytes: int = Field(default=DEFAULT_MAX_FILE_SIZE_BYTES, gt=0)
    default_ignore_patterns: tuple[str, ...] = DEFAULT_IGNORE_PATTERNS
    supported_extensions: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_SUPPORTED_EXTENSIONS)
    )

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
