"""Index contracts."""

from abc import ABC, abstractmethod

from codebase_rag.domain.models import Chunk, SearchHit


class VectorIndex(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, vector: list[float], limit: int) -> list[SearchHit]:
        raise NotImplementedError


class LexicalIndex(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, limit: int) -> list[SearchHit]:
        raise NotImplementedError
