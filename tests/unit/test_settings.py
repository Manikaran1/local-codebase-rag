import pytest
from pydantic import ValidationError

from codebase_rag.config.settings import Settings


def test_default_configuration_is_local_first() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url.startswith("http://localhost")
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_allow_download is False
    assert settings.vector_backend == "sqlite"
    assert settings.index_db_path == settings.data_dir / "indexes" / "index.sqlite3"
    assert settings.bm25_k1 == 1.5
    assert settings.bm25_b == 0.75


def test_ingestion_defaults_are_present_and_limits_are_validated() -> None:
    settings = Settings(_env_file=None)

    assert settings.max_file_size_bytes == 2 * 1024 * 1024
    assert settings.max_chunk_chars == 4000
    assert ".git/" in settings.default_ignore_patterns
    assert settings.supported_extensions[".py"] == "Python"
    assert settings.supported_extensions[".md"] == "Markdown"
    assert settings.supported_extensions[".txt"] == "text"

    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_file_size_bytes=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_chunk_chars=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, embedding_batch_size=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, bm25_k1=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, bm25_b=1.1)


def test_chunk_size_can_be_configured_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEBASE_RAG_MAX_CHUNK_CHARS", "250")

    assert Settings(_env_file=None).max_chunk_chars == 250


def test_embedding_download_policy_can_be_configured_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEBASE_RAG_EMBEDDING_ALLOW_DOWNLOAD", "true")

    assert Settings(_env_file=None).embedding_allow_download is True
