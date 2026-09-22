"""Citation-aware prompt context construction."""

from codebase_rag.domain.models import SearchHit


class ContextBuilder:
    def build(self, question: str, hits: list[SearchHit]) -> str:
        raise NotImplementedError
