"""Deterministic score fusion for vector and lexical retrieval candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from codebase_rag.domain.models import Chunk


@dataclass(frozen=True)
class FusionCandidate:
    """A merged candidate with independent channel ranks and raw scores."""

    chunk: Chunk
    vector_rank: int | None
    lexical_rank: int | None
    vector_score: float | None
    lexical_score: float | None
    normalized_vector_score: float
    normalized_lexical_score: float
    fusion_score: float
    rerank_score: float | None = None


class FusionStrategy(Protocol):
    """Narrow contract implemented by supported retrieval score fusers."""

    @property
    def name(self) -> str: ...

    def score(self, candidate: FusionCandidate, *, max_lexical_score: float) -> float: ...


@dataclass(frozen=True)
class WeightedFusion:
    """Weighted sum of query-local normalized cosine and BM25 scores."""

    vector_weight: float = 0.5
    lexical_weight: float = 0.5

    def __post_init__(self) -> None:
        _validate_weights(self.vector_weight, self.lexical_weight)

    @property
    def name(self) -> str:
        return "weighted"

    def score(self, candidate: FusionCandidate, *, max_lexical_score: float) -> float:
        total_weight = self.vector_weight + self.lexical_weight
        vector_weight = self.vector_weight / total_weight
        lexical_weight = self.lexical_weight / total_weight
        return (
            vector_weight * candidate.normalized_vector_score
            + lexical_weight * candidate.normalized_lexical_score
        )


@dataclass(frozen=True)
class ReciprocalRankFusion:
    """Weighted Reciprocal Rank Fusion using each index's independent rank."""

    vector_weight: float = 0.5
    lexical_weight: float = 0.5
    rrf_k: int = 60

    def __post_init__(self) -> None:
        _validate_weights(self.vector_weight, self.lexical_weight)
        if isinstance(self.rrf_k, bool) or not isinstance(self.rrf_k, int) or self.rrf_k <= 0:
            raise ValueError("rrf_k must be an integer greater than zero")

    @property
    def name(self) -> str:
        return "rrf"

    def score(self, candidate: FusionCandidate, *, max_lexical_score: float) -> float:
        score = 0.0
        if candidate.vector_rank is not None:
            score += self.vector_weight / (self.rrf_k + candidate.vector_rank)
        if candidate.lexical_rank is not None:
            score += self.lexical_weight / (self.rrf_k + candidate.lexical_rank)
        return score


def normalized_vector_score(raw_score: float | None) -> float:
    """Clamp raw cosine similarity into a non-negative ranking contribution."""
    if raw_score is None:
        return 0.0
    return min(1.0, max(0.0, raw_score))


def normalized_lexical_score(raw_score: float | None, max_score: float) -> float:
    """Scale a raw BM25 score against the strongest lexical hit for this query."""
    if raw_score is None or max_score <= 0:
        return 0.0
    return raw_score / max_score


def _validate_weights(vector_weight: float, lexical_weight: float) -> None:
    if (
        not math.isfinite(vector_weight)
        or not math.isfinite(lexical_weight)
        or vector_weight < 0
        or lexical_weight < 0
    ):
        raise ValueError("fusion weights must be non-negative finite values")
    if vector_weight + lexical_weight <= 0:
        raise ValueError("sum of fusion weights must be greater than zero")
