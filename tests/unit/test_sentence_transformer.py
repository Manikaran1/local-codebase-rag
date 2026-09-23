from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from codebase_rag.embeddings.errors import EmbeddingDependencyError, EmbeddingModelError
from codebase_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder


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

    embedder = SentenceTransformerEmbedder("local-model", batch_size=12, model_factory=factory)

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
    assert factory_calls == [("local-model", {"device": "cpu", "local_files_only": True})]
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

    embedder = SentenceTransformerEmbedder("private-model", model_factory=factory)

    with pytest.raises(EmbeddingModelError, match="local model directory") as error:
        embedder.embed_query("query")

    assert factory_options == [{"device": "cpu", "local_files_only": True}]
    assert "CODEBASE_RAG_EMBEDDING_ALLOW_DOWNLOAD=true" in str(error.value)


def test_download_is_only_enabled_by_explicit_opt_in() -> None:
    factory_options: list[dict[str, object]] = []
    model = _StubModel()

    def factory(model_name: str, **kwargs: object) -> _StubModel:
        factory_options.append(kwargs)
        return model

    embedder = SentenceTransformerEmbedder("model-id", allow_download=True, model_factory=factory)

    embedder.embed_query("query")

    assert factory_options == [{"device": "cpu", "local_files_only": False}]


def test_mismatched_output_count_fails_clearly() -> None:
    class BrokenModel(_StubModel):
        def encode_document(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            return []

    embedder = SentenceTransformerEmbedder(
        model_factory=lambda _model_name, **_: BrokenModel()
    )

    with pytest.raises(EmbeddingModelError, match="0 vectors for 1 inputs"):
        embedder.embed_documents(["text"])
