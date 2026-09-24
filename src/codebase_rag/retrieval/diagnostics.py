"""Structured diagnostics owned by hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CandidateSelectionStatus = Literal[
    "selected",
    "backfilled",
    "suppressed_overlap",
    "deferred_file_diversity",
    "not_selected_limit",
]


@dataclass(frozen=True)
class RetrievalCandidateTrace:
    """Channel evidence and selection outcome for one merged chunk candidate."""

    chunk_id: str
    vector_rank: int | None
    lexical_rank: int | None
    raw_vector_score: float | None
    raw_lexical_score: float | None
    normalized_vector_score: float
    normalized_lexical_score: float
    fusion_score: float
    rerank_score: float | None
    final_rank: int | None
    selection_status: CandidateSelectionStatus
    selection_reason: str | None = None


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Inspectable counts, configuration, and per-candidate ranking trace."""

    project_id: str
    normalized_query: str
    fusion_strategy: str
    vector_candidate_count: int
    lexical_candidate_count: int
    merged_candidate_count: int
    candidate_limit: int
    configured_candidate_limit: int
    final_limit: int
    reranker_applied: bool
    overlap_suppression_count: int
    diversity_deferral_count: int
    diversity_backfill_count: int
    duplicate_backend_hit_count: int
    filtered_foreign_project_count: int
    candidate_traces: tuple[RetrievalCandidateTrace, ...]
