"""Index contracts."""

from abc import ABC, abstractmethod

from codebase_rag.domain.models import Chunk, SearchHit


class VectorIndex(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, project_id: str, vector: list[float], limit: int) -> list[SearchHit]:
        raise NotImplementedError

    @abstractmethod
    def delete_file(self, project_id: str, file_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def delete_project(self, project_id: str) -> int:
        raise NotImplementedError


class LexicalIndex(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, project_id: str, query: str, limit: int) -> list[SearchHit]:
        raise NotImplementedError

    @abstractmethod
    def delete_file(self, project_id: str, file_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def delete_project(self, project_id: str) -> int:
        raise NotImplementedError
