"""Hybrid retrieval orchestration."""

from codebase_rag.domain.models import SearchHit


class HybridRetriever:
    """Combine semantic and lexical evidence.

    Implementation intentionally deferred until both retrieval baselines exist.
    """

    def retrieve(self, query: str, limit: int = 10) -> list[SearchHit]:
        raise NotImplementedError
