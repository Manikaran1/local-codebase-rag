"""Parser contracts."""

from abc import ABC, abstractmethod

from codebase_rag.domain.models import SourceFile, Symbol


class CodeParser(ABC):
    @abstractmethod
    def supports(self, source: SourceFile) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse(self, source: SourceFile, text: str) -> list[Symbol]:
        raise NotImplementedError
