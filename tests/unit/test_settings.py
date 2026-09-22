from codebase_rag.config.settings import Settings


def test_default_configuration_is_local_first() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url.startswith("http://localhost")
