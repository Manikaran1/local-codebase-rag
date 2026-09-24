from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from codebase_rag.embeddings.errors import EmbeddingDependencyError, EmbeddingModelError
from codebase_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder

_MODEL_REVISION = "0123456789abcdef0123456789abcdef01234567"


class _StubModel:
    def __init__(self) -> None:
        self.document_calls: list[tuple[list[str], dict[str, object]]] = []
        self.query_calls: list[tuple[list[str], dict[str, object]]] = []
        self.eval_calls = 0

    def eval(self) -> _StubModel:
        self.eval_calls += 1
        return self

    def encode_document(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.document_calls.append((texts, kwargs))
        return [[3.0, 4.0], [0.0, 2.0]][: len(texts)]

    def encode_query(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.query_calls.append((texts, kwargs))
        return [[0.0, 3.0] for _ in texts]


def test_sentence_transformer_loads_lazily_batches_and_normalizes() -> None:
    model = _StubModel()
    factory_calls: list[tuple[str, dict[str, object]]] = []

    def factory(model_name: str, **kwargs: object) -> _StubModel:
        factory_calls.append((model_name, kwargs))
        return model

    embedder = SentenceTransformerEmbedder(
        "sentence-transformers/test-model",
        revision=_MODEL_REVISION,
        batch_size=12,
        model_factory=factory,
    )

    assert factory_calls == []
    vectors = embedder.embed_documents(["first", "second"])
    query = embedder.embed_query("query")

    assert vectors == [[0.6, 0.8], [0.0, 1.0]]
    assert query == [0.0, 1.0]
    assert model.document_calls[0][0] == ["first", "second"]
    assert model.query_calls[0][0] == ["query"]
    for _, options in [model.document_calls[0], model.query_calls[0]]:
        assert options["normalize_embeddings"] is True
        assert options["batch_size"] == 12
        assert options["convert_to_numpy"] is True
        assert options["show_progress_bar"] is False
        assert options["device"] == "cpu"
    assert factory_calls == [
        (
            "sentence-transformers/test-model",
            {"device": "cpu", "local_files_only": True, "revision": _MODEL_REVISION},
        )
    ]
    assert embedder.spec.revision == _MODEL_REVISION
    assert model.eval_calls == 1


def test_empty_document_batch_does_not_load_model() -> None:
    def unexpected_factory(**kwargs: object) -> _StubModel:
        raise AssertionError(f"model should remain lazy: {kwargs}")

    embedder = SentenceTransformerEmbedder(model_factory=unexpected_factory)

    assert embedder.embed_documents([]) == []


def test_optional_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_dependency() -> Callable[..., Any]:
        raise EmbeddingDependencyError("Install local-embeddings extra")

    monkeypatch.setattr(
        "codebase_rag.embeddings.sentence_transformer._sentence_transformer_class",
        missing_dependency,
    )
    embedder = SentenceTransformerEmbedder()

    with pytest.raises(EmbeddingDependencyError, match="local-embeddings"):
        embedder.embed_query("query")


def test_offline_model_failure_explains_local_path_or_explicit_download() -> None:
    factory_options: list[dict[str, object]] = []

    def factory(model_name: str, **kwargs: object) -> _StubModel:
        factory_options.append(kwargs)
        raise OSError("model not cached")

    embedder = SentenceTransformerEmbedder(
        "private-model", revision=_MODEL_REVISION, model_factory=factory
    )

    with pytest.raises(EmbeddingModelError, match="local model directory") as error:
        embedder.embed_query("query")

    assert factory_options == [
        {"device": "cpu", "local_files_only": True, "revision": _MODEL_REVISION}
    ]
    assert "CODEBASE_RAG_EMBEDDING_ALLOW_DOWNLOAD=true" in str(error.value)


def test_download_is_only_enabled_by_explicit_opt_in() -> None:
    factory_options: list[dict[str, object]] = []
    model = _StubModel()

    def factory(model_name: str, **kwargs: object) -> _StubModel:
        factory_options.append(kwargs)
        return model

    embedder = SentenceTransformerEmbedder(
        "model-id",
        revision=_MODEL_REVISION,
        allow_download=True,
        model_factory=factory,
    )

    embedder.embed_query("query")

    assert factory_options == [
        {"device": "cpu", "local_files_only": False, "revision": _MODEL_REVISION}
    ]


def test_mismatched_output_count_fails_clearly() -> None:
    class BrokenModel(_StubModel):
        def encode_document(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            return []

    embedder = SentenceTransformerEmbedder(
        model_factory=lambda _model_name, **_: BrokenModel()
    )

    with pytest.raises(EmbeddingModelError, match="0 vectors for 1 inputs"):
        embedder.embed_documents(["text"])


def test_default_revision_is_pinned_and_custom_model_requires_revision(tmp_path) -> None:
    default_embedder = SentenceTransformerEmbedder()
    assert default_embedder.spec.revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

    with pytest.raises(ValueError, match="explicit immutable revision"):
        SentenceTransformerEmbedder("custom/model")
    with pytest.raises(ValueError, match="immutable 40-character commit SHAs"):
        SentenceTransformerEmbedder("custom/model", revision="main")

    local_model = tmp_path / "model"
    local_model.mkdir()
    with pytest.raises(ValueError, match="explicit revision or fingerprint"):
        SentenceTransformerEmbedder(str(local_model))


def test_local_model_uses_explicit_fingerprint_without_hub_revision(tmp_path) -> None:
    local_model = tmp_path / "model"
    local_model.mkdir()
    factory_options: list[dict[str, object]] = []

    def factory(model_name: str, **kwargs: object) -> _StubModel:
        factory_options.append(kwargs)
        return _StubModel()

    embedder = SentenceTransformerEmbedder(
        str(local_model), revision="sha256:content-fingerprint", model_factory=factory
    )

    assert embedder.spec.revision == "sha256:content-fingerprint"
    embedder.embed_query("query")
    assert factory_options == [{"device": "cpu", "local_files_only": True}]
