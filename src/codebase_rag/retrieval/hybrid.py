"""Project-scoped hybrid retrieval orchestration and evidence diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from codebase_rag.config.settings import Settings
from codebase_rag.domain.models import Chunk, SearchHit
from codebase_rag.embeddings.base import Embedder
from codebase_rag.index.base import LexicalIndex, VectorIndex
from codebase_rag.retrieval.diagnostics import (
    CandidateSelectionStatus,
    RetrievalCandidateTrace,
    RetrievalDiagnostics,
)
from codebase_rag.retrieval.fusion import (
    FusionCandidate,
    FusionStrategy,
    ReciprocalRankFusion,
    WeightedFusion,
    normalized_lexical_score,
    normalized_vector_score,
)
from codebase_rag.retrieval.rerank import NoOpReranker, Reranker

FusionStrategyName = Literal["weighted", "rrf"]


@dataclass(frozen=True)
class RetrievalResult:
    """Final evidence plus structured retrieval-owned diagnostics."""

    hits: tuple[SearchHit, ...]
    diagnostics: RetrievalDiagnostics


@dataclass
class _CandidateState:
    chunk_variants: list[Chunk] = field(default_factory=list)
    vector_rank: int | None = None
    lexical_rank: int | None = None
    vector_score: float | None = None
    lexical_score: float | None = None


class HybridRetriever:
    """Combine project-scoped vector and BM25 candidates behind index interfaces."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_index: VectorIndex,
        lexical_index: LexicalIndex,
        candidate_limit: int = 40,
        fusion_strategy: FusionStrategyName | FusionStrategy = "weighted",
        vector_weight: float = 0.5,
        lexical_weight: float = 0.5,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
        max_hits_per_file: int = 3,
        overlap_line_threshold: float = 0.8,
        overlap_text_threshold: float = 0.8,
    ) -> None:
        _validate_positive_integer(candidate_limit, "candidate_limit")
        _validate_positive_integer(max_hits_per_file, "max_hits_per_file")
        _validate_threshold(overlap_line_threshold, "overlap_line_threshold")
        _validate_threshold(overlap_text_threshold, "overlap_text_threshold")
        # Keep constructor validation consistent even when a custom strategy is injected.
        WeightedFusion(vector_weight, lexical_weight)
        _validate_positive_integer(rrf_k, "rrf_k")

        if isinstance(fusion_strategy, str):
            if fusion_strategy == "weighted":
                resolved_fusion: FusionStrategy = WeightedFusion(vector_weight, lexical_weight)
            elif fusion_strategy == "rrf":
                resolved_fusion = ReciprocalRankFusion(vector_weight, lexical_weight, rrf_k)
            else:
                raise ValueError(f"unsupported fusion strategy: {fusion_strategy!r}")
        else:
            resolved_fusion = fusion_strategy

        self.embedder = embedder
        self.vector_index = vector_index
        self.lexical_index = lexical_index
        self.candidate_limit = candidate_limit
        self.fusion_strategy = resolved_fusion
        self.reranker = reranker if reranker is not None else NoOpReranker()
        self.max_hits_per_file = max_hits_per_file
        self.overlap_line_threshold = overlap_line_threshold
        self.overlap_text_threshold = overlap_text_threshold

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        embedder: Embedder,
        vector_index: VectorIndex,
        lexical_index: LexicalIndex,
        reranker: Reranker | None = None,
    ) -> HybridRetriever:
        """Build a retriever from Phase 5 settings and explicit dependencies."""
        return cls(
            embedder=embedder,
            vector_index=vector_index,
            lexical_index=lexical_index,
            candidate_limit=settings.retrieval_candidate_limit,
            fusion_strategy=settings.retrieval_fusion_strategy,
            vector_weight=settings.retrieval_vector_weight,
            lexical_weight=settings.retrieval_lexical_weight,
            rrf_k=settings.retrieval_rrf_k,
            reranker=reranker,
            max_hits_per_file=settings.retrieval_max_hits_per_file,
            overlap_line_threshold=settings.retrieval_overlap_line_threshold,
            overlap_text_threshold=settings.retrieval_overlap_text_threshold,
        )

    def retrieve(self, project_id: str, query: str, limit: int = 10) -> RetrievalResult:
        """Return final ranked evidence and a trace of every merged candidate."""
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("project_id must not be empty")
        _validate_positive_integer(limit, "limit")
        normalized_query = normalize_query(query)
        effective_candidate_limit = max(limit, self.candidate_limit)

        query_vector = self.embedder.embed_query(normalized_query)
        vector_hits = self.vector_index.search(
            project_id, query_vector, effective_candidate_limit
        )
        lexical_hits = self.lexical_index.search(
            project_id, normalized_query, effective_candidate_limit
        )

        states: dict[str, _CandidateState] = {}
        duplicate_count = 0
        foreign_count = 0
        duplicate_count += self._merge_channel(
            states,
            vector_hits,
            project_id=project_id,
            score_field="vector_score",
            rank_field="vector_rank",
        )
        duplicate_count += self._merge_channel(
            states,
            lexical_hits,
            project_id=project_id,
            score_field="lexical_score",
            rank_field="lexical_rank",
        )
        foreign_count = sum(hit.chunk.project_id != project_id for hit in vector_hits)
        foreign_count += sum(hit.chunk.project_id != project_id for hit in lexical_hits)

        max_lexical_score = max(
            (state.lexical_score for state in states.values() if state.lexical_score is not None),
            default=0.0,
        )
        candidates = [
            self._make_candidate(chunk_id, state, max_lexical_score)
            for chunk_id, state in states.items()
        ]
        candidates.sort(key=lambda item: (-item.fusion_score, item.chunk.id))

        reranker_applied = bool(self.reranker.is_active and candidates)
        if reranker_applied:
            candidates = self._apply_reranker(normalized_query, candidates)
            candidates.sort(
                key=lambda item: (
                    -_required_score(item.rerank_score),
                    -item.fusion_score,
                    item.chunk.id,
                )
            )

        survivors, selection_by_id, overlap_suppression_count = self._suppress_redundancy(
            candidates
        )
        selected, diversity_deferred_count, diversity_backfill_count = self._select_diverse(
            survivors,
            selection_by_id,
            limit,
        )

        final_rank_by_id = {candidate.chunk.id: rank for rank, candidate in enumerate(selected, 1)}
        hits = tuple(
            SearchHit(
                chunk=candidate.chunk,
                rank=rank,
                vector_score=candidate.vector_score,
                lexical_score=candidate.lexical_score,
                fusion_score=candidate.fusion_score,
                rerank_score=candidate.rerank_score,
            )
            for rank, candidate in enumerate(selected, 1)
        )
        traces = tuple(
            RetrievalCandidateTrace(
                chunk_id=candidate.chunk.id,
                vector_rank=candidate.vector_rank,
                lexical_rank=candidate.lexical_rank,
                raw_vector_score=candidate.vector_score,
                raw_lexical_score=candidate.lexical_score,
                normalized_vector_score=candidate.normalized_vector_score,
                normalized_lexical_score=candidate.normalized_lexical_score,
                fusion_score=candidate.fusion_score,
                rerank_score=candidate.rerank_score,
                final_rank=final_rank_by_id.get(candidate.chunk.id),
                selection_status=selection_by_id[candidate.chunk.id][0],
                selection_reason=selection_by_id[candidate.chunk.id][1],
            )
            for candidate in candidates
        )
        diagnostics = RetrievalDiagnostics(
            project_id=project_id,
            normalized_query=normalized_query,
            fusion_strategy=self.fusion_strategy.name,
            vector_candidate_count=len(vector_hits),
            lexical_candidate_count=len(lexical_hits),
            merged_candidate_count=len(states),
            candidate_limit=effective_candidate_limit,
            configured_candidate_limit=self.candidate_limit,
            final_limit=limit,
            reranker_applied=reranker_applied,
            overlap_suppression_count=overlap_suppression_count,
            diversity_deferral_count=diversity_deferred_count,
            diversity_backfill_count=diversity_backfill_count,
            duplicate_backend_hit_count=duplicate_count,
            filtered_foreign_project_count=foreign_count,
            candidate_traces=traces,
        )
        return RetrievalResult(hits=hits, diagnostics=diagnostics)

    def _merge_channel(
        self,
        states: dict[str, _CandidateState],
        hits: Sequence[SearchHit],
        *,
        project_id: str,
        score_field: Literal["vector_score", "lexical_score"],
        rank_field: Literal["vector_rank", "lexical_rank"],
    ) -> int:
        duplicate_count = 0
        for hit in hits:
            if hit.chunk.project_id != project_id:
                continue
            if (
                isinstance(hit.rank, bool)
                or not isinstance(hit.rank, int)
                or hit.rank <= 0
            ):
                raise ValueError("index candidate ranks must be positive integers")
            raw_score = getattr(hit, score_field)
            if raw_score is None:
                raise ValueError(f"index candidate is missing its raw {score_field}")
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError(f"index candidate {score_field} must be finite")

            state = states.setdefault(hit.chunk.id, _CandidateState())
            state.chunk_variants.append(hit.chunk)
            old_score = getattr(state, score_field)
            old_rank = getattr(state, rank_field)
            if old_score is not None:
                duplicate_count += 1
            if old_score is None or score > old_score:
                setattr(state, score_field, score)
            if old_rank is None or hit.rank < old_rank:
                setattr(state, rank_field, hit.rank)
        return duplicate_count

    def _make_candidate(
        self,
        chunk_id: str,
        state: _CandidateState,
        max_lexical_score: float,
    ) -> FusionCandidate:
        chunk = min(state.chunk_variants, key=_chunk_sort_key)
        candidate = FusionCandidate(
            chunk=chunk,
            vector_rank=state.vector_rank,
            lexical_rank=state.lexical_rank,
            vector_score=state.vector_score,
            lexical_score=state.lexical_score,
            normalized_vector_score=normalized_vector_score(state.vector_score),
            normalized_lexical_score=normalized_lexical_score(
                state.lexical_score, max_lexical_score
            ),
            fusion_score=0.0,
        )
        fused_score = self.fusion_strategy.score(
            candidate, max_lexical_score=max_lexical_score
        )
        if not math.isfinite(fused_score):
            raise ValueError(f"fusion strategy produced a non-finite score for {chunk_id!r}")
        return replace(candidate, fusion_score=float(fused_score))

    def _apply_reranker(
        self,
        query: str,
        candidates: list[FusionCandidate],
    ) -> list[FusionCandidate]:
        scores = self.reranker.rerank(query, tuple(candidates))
        expected_ids = {candidate.chunk.id for candidate in candidates}
        if not isinstance(scores, Mapping) or set(scores) != expected_ids:
            raise ValueError("reranker must return exactly one score for every candidate chunk_id")
        reranked: list[FusionCandidate] = []
        for candidate in candidates:
            score = scores[candidate.chunk.id]
            if isinstance(score, bool):
                raise ValueError("reranker scores must be finite numbers")
            try:
                numeric_score = float(score)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("reranker scores must be finite numbers") from error
            if not math.isfinite(numeric_score):
                raise ValueError("reranker scores must be finite numbers")
            reranked.append(replace(candidate, rerank_score=numeric_score))
        return reranked

    def _suppress_redundancy(
        self,
        candidates: Sequence[FusionCandidate],
    ) -> tuple[
        list[FusionCandidate],
        dict[str, tuple[CandidateSelectionStatus, str | None]],
        int,
    ]:
        survivors: list[FusionCandidate] = []
        selection_by_id: dict[str, tuple[CandidateSelectionStatus, str | None]] = {
            candidate.chunk.id: ("not_selected_limit", None) for candidate in candidates
        }
        suppression_count = 0
        for candidate in candidates:
            redundant_with = next(
                (
                    kept
                    for kept in survivors
                    if _is_substantially_redundant(
                        candidate.chunk,
                        kept.chunk,
                        line_threshold=self.overlap_line_threshold,
                        text_threshold=self.overlap_text_threshold,
                    )
                ),
                None,
            )
            if redundant_with is None:
                survivors.append(candidate)
            else:
                suppression_count += 1
                selection_by_id[candidate.chunk.id] = (
                    "suppressed_overlap",
                    f"redundant_with:{redundant_with.chunk.id}",
                )
        return survivors, selection_by_id, suppression_count

    def _select_diverse(
        self,
        candidates: Sequence[FusionCandidate],
        selection_by_id: dict[str, tuple[CandidateSelectionStatus, str | None]],
        limit: int,
    ) -> tuple[list[FusionCandidate], int, int]:
        selected: list[FusionCandidate] = []
        deferred: list[FusionCandidate] = []
        counts_by_file: Counter[str] = Counter()

        for candidate in candidates:
            file_id = candidate.chunk.file_id
            if counts_by_file[file_id] >= self.max_hits_per_file:
                deferred.append(candidate)
                selection_by_id[candidate.chunk.id] = (
                    "deferred_file_diversity",
                    f"file_limit_reached:{file_id}",
                )
            elif len(selected) < limit:
                selected.append(candidate)
                counts_by_file[file_id] += 1
                selection_by_id[candidate.chunk.id] = ("selected", "file_diversity_first_pass")
            else:
                selection_by_id[candidate.chunk.id] = ("not_selected_limit", "final_limit_reached")

        backfilled = 0
        for candidate in deferred:
            if len(selected) >= limit:
                continue
            selected.append(candidate)
            backfilled += 1
            selection_by_id[candidate.chunk.id] = (
                "backfilled",
                "backfilled_after_file_diversity",
            )
        return selected, len(deferred), backfilled


def normalize_query(query: str) -> str:
    """Trim and collapse whitespace without changing case or query syntax."""
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    return normalized


def _is_substantially_redundant(
    candidate: Chunk,
    kept: Chunk,
    *,
    line_threshold: float,
    text_threshold: float,
) -> bool:
    if candidate.file_id != kept.file_id:
        return False
    shorter_span = min(
        candidate.span.end_line - candidate.span.start_line + 1,
        kept.span.end_line - kept.span.start_line + 1,
    )
    if shorter_span <= 0:
        return False
    shared_lines = max(
        0,
        min(candidate.span.end_line, kept.span.end_line)
        - max(candidate.span.start_line, kept.span.start_line)
        + 1,
    )
    if shared_lines / shorter_span < line_threshold:
        return False

    candidate_text = " ".join(candidate.text.split())
    kept_text = " ".join(kept.text.split())
    if not candidate_text or not kept_text:
        return False
    shorter, longer = sorted((candidate_text, kept_text), key=lambda value: (len(value), value))
    return shorter in longer and len(shorter) / len(longer) >= text_threshold


def _chunk_sort_key(chunk: Chunk) -> tuple[object, ...]:
    metadata = repr(sorted((str(key), repr(value)) for key, value in chunk.metadata.items()))
    return (
        chunk.project_id,
        chunk.file_id,
        chunk.relative_path.as_posix(),
        chunk.span.start_line,
        chunk.span.end_line,
        chunk.text,
        chunk.language or "",
        chunk.symbol_id or "",
        chunk.symbol_name or "",
        chunk.symbol_kind or "",
        chunk.content_hash,
        metadata,
    )


def _required_score(score: float | None) -> float:
    if score is None:
        raise ValueError("reranked candidate is missing its rerank score")
    return score


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer greater than zero")


def _validate_threshold(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
