import pytest
from pydantic import ValidationError

from codebase_rag.config.settings import Settings


def test_default_configuration_is_local_first() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url.startswith("http://localhost")


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


def test_chunk_size_can_be_configured_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEBASE_RAG_MAX_CHUNK_CHARS", "250")

    assert Settings(_env_file=None).max_chunk_chars == 250
