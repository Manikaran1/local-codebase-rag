"""CPU Sentence Transformers adapter with offline-first model loading."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any, cast

from codebase_rag.config.settings import DEFAULT_EMBEDDING_MODEL
from codebase_rag.domain.models import EmbeddingSpec
from codebase_rag.embeddings.base import Embedder
from codebase_rag.embeddings.errors import EmbeddingDependencyError, EmbeddingModelError


def _sentence_transformer_class() -> Callable[..., Any]:
    try:
        module = import_module("sentence_transformers")
    except ImportError as error:
        raise EmbeddingDependencyError(
            "Sentence Transformers support is optional. Install it with "
            '`pip install -e ".[local-embeddings]"` and make the model available locally.'
        ) from error
    return cast(Callable[..., Any], module.SentenceTransformer)


class SentenceTransformerEmbedder(Embedder):
    """Lazily load and normalize local sentence-transformer embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        allow_download: bool = False,
        batch_size: int = 32,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self._model_name = model_name
        self._allow_download = allow_download
        self._batch_size = batch_size
        self._model_factory = model_factory
        self._model: Any | None = None
        self._model_ready = False

    @property
    def spec(self) -> EmbeddingSpec:
        return EmbeddingSpec(
            provider="sentence-transformers",
            model=self._model_name,
            normalized=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, query=False)

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("query text must be a string")
        vectors = self._encode([text], query=True)
        return vectors[0]

    def _encode(self, texts: Sequence[str], *, query: bool) -> list[list[float]]:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("all embedding inputs must be strings")
        if not texts:
            return []

        model = self._load_model()
        encode = model.encode_query if query else model.encode_document
        encoded = encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device="cpu",
        )
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        try:
            rows = list(encoded)
        except TypeError as error:
            raise EmbeddingModelError("The embedding model returned an invalid result.") from error
        if len(rows) != len(texts):
            raise EmbeddingModelError(
                f"The embedding model returned {len(rows)} vectors for {len(texts)} inputs."
            )

        vectors: list[list[float]] = []
        dimension: int | None = None
        for row in rows:
            if hasattr(row, "tolist"):
                row = row.tolist()
            try:
                vector = [float(value) for value in row]
            except (TypeError, ValueError) as error:
                raise EmbeddingModelError(
                    "The embedding model returned a non-numeric vector."
                ) from error
            if not vector or any(not math.isfinite(value) for value in vector):
                raise EmbeddingModelError(
                    "The embedding model returned an empty or non-finite vector."
                )
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise EmbeddingModelError(
                    "The embedding model returned vectors with mixed dimensions."
                )
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:
                raise EmbeddingModelError("The embedding model returned a zero vector.")
            vectors.append([value / norm for value in vector])
        return vectors

    def _load_model(self) -> Any:
        if self._model_ready:
            return self._model

        factory = self._model_factory or _sentence_transformer_class()
        if self._model is None:
            try:
                self._model = factory(
                    self._model_name,
                    device="cpu",
                    local_files_only=not self._allow_download,
                )
            except Exception as error:
                mode = "download-enabled mode" if self._allow_download else "offline mode"
                action = (
                    "check the model name and download connectivity, or set "
                    "CODEBASE_RAG_EMBEDDING_ALLOW_DOWNLOAD=true to acquire model artifacts"
                    if self._allow_download
                    else (
                        "set CODEBASE_RAG_EMBEDDING_MODEL to a local model directory, "
                        "cache the model before indexing, or explicitly set "
                        "CODEBASE_RAG_EMBEDDING_ALLOW_DOWNLOAD=true"
                    )
                )
                raise EmbeddingModelError(
                    f"Could not load embedding model {self._model_name!r} in {mode}; {action}. "
                    "Repository text is never sent to a remote embedding API."
                ) from error

        evaluate = getattr(self._model, "eval", None)
        if callable(evaluate):
            evaluate()
        self._model_ready = True
        return self._model
