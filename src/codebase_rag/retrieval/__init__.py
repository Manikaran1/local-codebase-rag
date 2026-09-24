"""Project-scoped retrieval, fusion, reranking, and diagnostic contracts."""

from codebase_rag.retrieval.diagnostics import RetrievalCandidateTrace, RetrievalDiagnostics
from codebase_rag.retrieval.fusion import (
    FusionCandidate,
    FusionStrategy,
    ReciprocalRankFusion,
    WeightedFusion,
)
from codebase_rag.retrieval.hybrid import HybridRetriever, RetrievalResult, normalize_query
from codebase_rag.retrieval.rerank import NoOpReranker, Reranker

__all__ = [
    "FusionCandidate",
    "FusionStrategy",
    "HybridRetriever",
    "NoOpReranker",
    "ReciprocalRankFusion",
    "Reranker",
    "RetrievalCandidateTrace",
    "RetrievalDiagnostics",
    "RetrievalResult",
    "WeightedFusion",
    "normalize_query",
]
