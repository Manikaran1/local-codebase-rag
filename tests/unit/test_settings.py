import pytest
from pydantic import ValidationError

from codebase_rag.config.settings import Settings
from codebase_rag.embeddings.factory import create_embedder


def test_default_configuration_is_local_first() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url.startswith("http://localhost")
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_revision is None
    assert settings.embedding_allow_download is False
    assert settings.vector_backend == "sqlite"
    assert settings.index_db_path == settings.data_dir / "indexes" / "index.sqlite3"
    assert settings.bm25_k1 == 1.5
    assert settings.bm25_b == 0.75
    assert settings.retrieval_candidate_limit == 40
    assert settings.retrieval_fusion_strategy == "weighted"
    assert settings.retrieval_vector_weight == 0.5
    assert settings.retrieval_lexical_weight == 0.5
    assert settings.retrieval_rrf_k == 60
    assert settings.retrieval_max_hits_per_file == 3
    assert settings.retrieval_overlap_line_threshold == 0.8
    assert settings.retrieval_overlap_text_threshold == 0.8


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

    for invalid in (
        {"retrieval_candidate_limit": 0},
        {"retrieval_fusion_strategy": "custom"},
        {"retrieval_vector_weight": -0.1},
        {"retrieval_vector_weight": 0.0, "retrieval_lexical_weight": 0.0},
        {"retrieval_rrf_k": 0},
        {"retrieval_max_hits_per_file": 0},
        {"retrieval_overlap_line_threshold": 1.1},
        {"retrieval_overlap_text_threshold": -0.1},
    ):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **invalid)


def test_chunk_size_can_be_configured_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEBASE_RAG_MAX_CHUNK_CHARS", "250")

    assert Settings(_env_file=None).max_chunk_chars == 250


def test_embedding_download_policy_can_be_configured_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEBASE_RAG_EMBEDDING_ALLOW_DOWNLOAD", "true")

    assert Settings(_env_file=None).embedding_allow_download is True


def test_retrieval_settings_can_be_configured_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEBASE_RAG_RETRIEVAL_CANDIDATE_LIMIT", "64")
    monkeypatch.setenv("CODEBASE_RAG_RETRIEVAL_FUSION_STRATEGY", "rrf")
    monkeypatch.setenv("CODEBASE_RAG_RETRIEVAL_MAX_HITS_PER_FILE", "2")

    settings = Settings(_env_file=None)

    assert settings.retrieval_candidate_limit == 64
    assert settings.retrieval_fusion_strategy == "rrf"
    assert settings.retrieval_max_hits_per_file == 2


def test_embedding_revision_can_be_configured_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setenv("CODEBASE_RAG_EMBEDDING_REVISION", revision)

    assert Settings(_env_file=None).embedding_revision == revision


def test_embedder_factory_resolves_default_and_configured_revisions() -> None:
    default_embedder = create_embedder(Settings(_env_file=None))
    assert default_embedder.spec.revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

    revision = "0123456789abcdef0123456789abcdef01234567"
    custom_embedder = create_embedder(
        Settings(
            _env_file=None,
            embedding_model="custom/model",
            embedding_revision=revision,
        )
    )
    assert custom_embedder.spec.revision == revision
