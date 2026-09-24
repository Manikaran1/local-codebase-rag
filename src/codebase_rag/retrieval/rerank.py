"""Optional deterministic reranker boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from codebase_rag.retrieval.fusion import FusionCandidate as RerankCandidate


class Reranker(Protocol):
    """Score fused candidates for a normalized query."""

    is_active: bool

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> Mapping[str, float]: ...


class NoOpReranker:
    """Default baseline; leaves fused ranking unchanged and adds no scores."""

    is_active = False

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> Mapping[str, float]:
        return {}
