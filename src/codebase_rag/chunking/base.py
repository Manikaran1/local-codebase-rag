"""Chunking contracts."""

from abc import ABC, abstractmethod

from codebase_rag.domain.models import Chunk, SourceFile, Symbol


class Chunker(ABC):
    @abstractmethod
    def chunk(self, source: SourceFile, text: str, symbols: list[Symbol]) -> list[Chunk]:
        raise NotImplementedError
