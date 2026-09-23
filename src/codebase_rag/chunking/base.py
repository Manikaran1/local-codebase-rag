"""Chunking contracts."""

from abc import ABC, abstractmethod

from codebase_rag.domain.models import Chunk, ParsedDocument, SourceFile, Symbol


class Chunker(ABC):
    @abstractmethod
    def chunk(self, source: SourceFile, text: str, symbols: list[Symbol]) -> list[Chunk]:
        raise NotImplementedError


class DocumentChunker(ABC):
    @abstractmethod
    def chunk(self, source: SourceFile, document: ParsedDocument) -> list[Chunk]:
        raise NotImplementedError
