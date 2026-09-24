"""Embedding provider contract."""

from abc import ABC, abstractmethod

from codebase_rag.domain.models import EmbeddingSpec


class Embedder(ABC):
    @property
    @abstractmethod
    def spec(self) -> EmbeddingSpec:
        """Describe the model settings that determine vector compatibility."""
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError
