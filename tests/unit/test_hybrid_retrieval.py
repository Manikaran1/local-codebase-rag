from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from codebase_rag.config.settings import Settings
from codebase_rag.domain.models import Chunk, EmbeddingSpec, SearchHit, SourceSpan
from codebase_rag.embeddings.base import Embedder
from codebase_rag.index.base import LexicalIndex, VectorIndex
from codebase_rag.retrieval.hybrid import HybridRetriever, RetrievalResult
from codebase_rag.retrieval.rerank import NoOpReranker, RerankCandidate, Reranker


def _chunk(
    chunk_id: str,
    text: str,
    *,
    file_id: str | None = None,
    path: str | None = None,
    project_id: str = "project-a",
    start: int = 1,
    end: int = 1,
) -> Chunk:
    resolved_file = file_id or f"{chunk_id}.py"
    return Chunk(
        id=chunk_id,
        project_id=project_id,
        file_id=resolved_file,
        relative_path=Path(path or resolved_file),
        text=text,
        span=SourceSpan(start_line=start, end_line=end),
    )


def _hit(
    chunk_id: str,
    text: str,
    *,
    rank: int,
    vector: float | None = None,
    lexical: float | None = None,
    file_id: str | None = None,
    path: str | None = None,
    project_id: str = "project-a",
    start: int = 1,
    end: int = 1,
) -> SearchHit:
    return SearchHit(
        chunk=_chunk(
            chunk_id,
            text,
            file_id=file_id,
            path=path,
            project_id=project_id,
            start=start,
            end=end,
        ),
        rank=rank,
        vector_score=vector,
        lexical_score=lexical,
    )


class FixedEmbedder(Embedder):
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.failure: Exception | None = None

    @property
    def spec(self) -> EmbeddingSpec:
        return EmbeddingSpec("test", "fixed", "v1", normalized=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        if self.failure is not None:
            raise self.failure
        return [1.0, 0.0]


class FixedVectorIndex(VectorIndex):
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = hits or []
        self.calls: list[tuple[str, list[float], int]] = []
        self.failure: Exception | None = None

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError

    def search(self, project_id: str, vector: list[float], limit: int) -> list[SearchHit]:
        self.calls.append((project_id, vector, limit))
        if self.failure is not None:
            raise self.failure
        return list(self.hits)

    def delete_file(self, project_id: str, file_id: str) -> int:
        raise NotImplementedError

    def delete_project(self, project_id: str) -> int:
        raise NotImplementedError


class FixedLexicalIndex(LexicalIndex):
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = hits or []
        self.calls: list[tuple[str, str, int]] = []
        self.failure: Exception | None = None

    def upsert(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def search(self, project_id: str, query: str, limit: int) -> list[SearchHit]:
        self.calls.append((project_id, query, limit))
        if self.failure is not None:
            raise self.failure
        return list(self.hits)

    def delete_file(self, project_id: str, file_id: str) -> int:
        raise NotImplementedError

    def delete_project(self, project_id: str) -> int:
        raise NotImplementedError


class DeterministicTestReranker(Reranker):
    is_active = True

    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = scores
        self.queries: list[str] = []
        self.candidate_ids: list[str] = []

    def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> Mapping[str, float]:
        self.queries.append(query)
        self.candidate_ids = [candidate.chunk.id for candidate in candidates]
        return self.scores


def _retriever(
    vector_hits: list[SearchHit] | None = None,
    lexical_hits: list[SearchHit] | None = None,
    **kwargs: object,
) -> tuple[HybridRetriever, FixedEmbedder, FixedVectorIndex, FixedLexicalIndex]:
    embedder = FixedEmbedder()
    vector = FixedVectorIndex(vector_hits)
    lexical = FixedLexicalIndex(lexical_hits)
    retriever = HybridRetriever(
        embedder=embedder,
        vector_index=vector,
        lexical_index=lexical,
        **kwargs,
    )
    return retriever, embedder, vector, lexical


def test_query_normalization_is_conservative_and_candidate_limit_is_effective() -> None:
    retriever, embedder, vector, lexical = _retriever(candidate_limit=4)

    result = retriever.retrieve(
        "project-a",
        "   RepositoryScanner   source_offset CODEBASE_RAG_EMBEDDING_MODEL, HTTP 401   ",
        limit=7,
    )

    expected = "RepositoryScanner source_offset CODEBASE_RAG_EMBEDDING_MODEL, HTTP 401"
    assert isinstance(result, RetrievalResult)
    assert embedder.queries == [expected]
    assert vector.calls == [("project-a", [1.0, 0.0], 7)]
    assert lexical.calls == [("project-a", expected, 7)]
    assert result.diagnostics.normalized_query == expected
    assert result.diagnostics.candidate_limit == 7
    assert result.diagnostics.configured_candidate_limit == 4
    assert result.diagnostics.final_limit == 7


def test_from_settings_uses_injected_indexes_and_phase_five_configuration() -> None:
    settings = Settings(
        _env_file=None,
        retrieval_candidate_limit=12,
        retrieval_fusion_strategy="rrf",
        retrieval_vector_weight=2.0,
        retrieval_lexical_weight=1.0,
        retrieval_rrf_k=9,
        retrieval_max_hits_per_file=2,
        retrieval_overlap_line_threshold=0.7,
        retrieval_overlap_text_threshold=0.6,
    )
    embedder = FixedEmbedder()
    vector = FixedVectorIndex()
    lexical = FixedLexicalIndex()

    retriever = HybridRetriever.from_settings(
        settings,
        embedder=embedder,
        vector_index=vector,
        lexical_index=lexical,
    )
    result = retriever.retrieve("project-a", "query", limit=4)

    assert result.diagnostics.fusion_strategy == "rrf"
    assert result.diagnostics.candidate_limit == 12
    assert vector.calls[0][2] == 12
    assert lexical.calls[0][2] == 12


def test_query_normalization_preserves_case_punctuation_and_identifiers() -> None:
    retriever, embedder, _, _ = _retriever()
    query = "where   is authenticate?  snake_case_identifier"

    retriever.retrieve("project-a", query)

    assert embedder.queries == ["where is authenticate? snake_case_identifier"]


def test_blank_query_project_and_limit_are_rejected_before_search() -> None:
    retriever, embedder, vector, lexical = _retriever()

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve("project-a", " \n\t ")
    with pytest.raises(ValueError, match="project_id must not be empty"):
        retriever.retrieve("  ", "valid")
    with pytest.raises(ValueError, match="limit must be an integer greater than zero"):
        retriever.retrieve("project-a", "valid", limit=0)
    with pytest.raises(ValueError, match="limit must be an integer greater than zero"):
        retriever.retrieve("project-a", "valid", limit=True)

    assert embedder.queries == []
    assert vector.calls == []
    assert lexical.calls == []


def test_weighted_fusion_merges_channels_and_preserves_raw_scores_and_ranks() -> None:
    vector_hits = [
        _hit("both", "shared evidence", rank=2, vector=0.8),
        _hit("vector-only", "semantic evidence", rank=1, vector=0.7),
    ]
    lexical_hits = [
        _hit("both", "shared evidence", rank=1, lexical=4.0),
        _hit("lexical-only", "exact identifier", rank=2, lexical=8.0),
    ]
    retriever, _, _, _ = _retriever(vector_hits, lexical_hits)

    result = retriever.retrieve("project-a", "evidence", limit=10)
    by_id = {hit.chunk.id: hit for hit in result.hits}
    traces = {trace.chunk_id: trace for trace in result.diagnostics.candidate_traces}

    assert result.diagnostics.vector_candidate_count == 2
    assert result.diagnostics.lexical_candidate_count == 2
    assert result.diagnostics.merged_candidate_count == 3
    assert by_id["both"].vector_score == 0.8
    assert by_id["both"].lexical_score == 4.0
    assert by_id["both"].fusion_score == pytest.approx(0.65)
    assert by_id["both"].rank == 1
    assert traces["both"].vector_rank == 2
    assert traces["both"].lexical_rank == 1
    assert traces["both"].raw_vector_score == 0.8
    assert traces["both"].raw_lexical_score == 4.0
    assert traces["both"].normalized_vector_score == 0.8
    assert traces["both"].normalized_lexical_score == 0.5
    assert traces["both"].fusion_score == pytest.approx(0.65)
    assert traces["vector-only"].vector_rank == 1
    assert traces["vector-only"].lexical_rank is None
    assert by_id["vector-only"].lexical_score is None
    assert traces["lexical-only"].vector_rank is None
    assert traces["lexical-only"].lexical_rank == 2
    assert by_id["lexical-only"].vector_score is None
    assert by_id["lexical-only"].lexical_score == 8.0


def test_weighted_fusion_clamps_cosine_and_supports_zero_or_unequal_weights() -> None:
    hits = [
        _hit("negative", "negative cosine", rank=1, vector=-0.4),
        _hit("vector-only", "vector", rank=2, vector=0.6),
    ]
    retriever, _, _, _ = _retriever(
        hits,
        [_hit("lexical-only", "lexical", rank=1, lexical=0.0)],
        vector_weight=3.0,
        lexical_weight=1.0,
    )

    result = retriever.retrieve("project-a", "anything")
    traces = {trace.chunk_id: trace for trace in result.diagnostics.candidate_traces}

    assert traces["negative"].normalized_vector_score == 0.0
    assert traces["negative"].fusion_score == 0.0
    assert traces["vector-only"].fusion_score == pytest.approx(0.75 * 0.6)
    assert traces["lexical-only"].normalized_lexical_score == 0.0
    assert traces["lexical-only"].fusion_score == 0.0


def test_rrf_uses_independent_channel_ranks_custom_k_weights_and_id_ties() -> None:
    vector_hits = [
        _hit("both", "both", rank=2, vector=0.2),
        _hit("vector-only", "vector", rank=1, vector=0.1),
        _hit("tie-z", "tie", rank=4, vector=0.0),
        _hit("tie-a", "tie", rank=4, vector=0.0),
    ]
    lexical_hits = [
        _hit("both", "both", rank=3, lexical=2.0),
        _hit("lexical-only", "lexical", rank=1, lexical=1.0),
    ]
    retriever, _, _, _ = _retriever(
        vector_hits,
        lexical_hits,
        fusion_strategy="rrf",
        vector_weight=2.0,
        lexical_weight=1.0,
        rrf_k=10,
    )

    result = retriever.retrieve("project-a", "combined", limit=10)
    by_id = {hit.chunk.id: hit for hit in result.hits}

    assert result.diagnostics.fusion_strategy == "rrf"
    assert by_id["both"].fusion_score == pytest.approx(2 / 12 + 1 / 13)
    assert by_id["vector-only"].fusion_score == pytest.approx(2 / 11)
    assert by_id["lexical-only"].fusion_score == pytest.approx(1 / 11)
    assert by_id["both"].vector_score == 0.2
    assert by_id["both"].lexical_score == 2.0
    assert by_id["tie-a"].rank < by_id["tie-z"].rank


def test_duplicate_vector_hits_keep_max_score_and_min_rank_independent_of_order() -> None:
    def run(hits: list[SearchHit]) -> RetrievalResult:
        retriever, _, _, _ = _retriever(hits, [])
        return retriever.retrieve("project-a", "duplicate")

    candidates = [
        _hit("same", "same text", rank=1, vector=0.6),
        _hit("same", "same text", rank=4, vector=0.9),
        _hit("same", "same text", rank=3, vector=0.9),
    ]
    first = run(candidates)
    second = run(list(reversed(candidates)))

    assert first == second
    assert len(first.hits) == 1
    assert first.hits[0].vector_score == 0.9
    trace = first.diagnostics.candidate_traces[0]
    assert trace.vector_rank == 1
    assert trace.raw_vector_score == 0.9
    assert first.diagnostics.duplicate_backend_hit_count == 2


def test_duplicate_lexical_hits_keep_max_score_and_min_rank_independently() -> None:
    retriever, _, _, _ = _retriever(
        [],
        [
            _hit("same", "same text", rank=1, lexical=0.6),
            _hit("same", "same text", rank=4, lexical=0.9),
            _hit("same", "same text", rank=3, lexical=0.9),
        ],
    )

    result = retriever.retrieve("project-a", "duplicate")

    assert len(result.hits) == 1
    assert result.hits[0].lexical_score == 0.9
    trace = result.diagnostics.candidate_traces[0]
    assert trace.lexical_rank == 1
    assert trace.raw_lexical_score == 0.9
    assert result.diagnostics.duplicate_backend_hit_count == 2


def test_rrf_duplicate_hit_uses_best_rank_independent_of_max_score() -> None:
    retriever, _, _, _ = _retriever(
        [
            _hit("a", "chunk A", rank=1, vector=0.5),
            _hit("a", "chunk A", rank=4, vector=0.9),
            _hit("b", "chunk B", rank=2, vector=0.8),
        ],
        [],
        fusion_strategy="rrf",
        vector_weight=1.0,
        lexical_weight=1.0,
        rrf_k=10,
    )

    result = retriever.retrieve("project-a", "duplicate", limit=10)
    by_id = {hit.chunk.id: hit for hit in result.hits}
    traces = {trace.chunk_id: trace for trace in result.diagnostics.candidate_traces}

    assert [hit.chunk.id for hit in result.hits[:2]] == ["a", "b"]
    assert by_id["a"].fusion_score == pytest.approx(1 / 11)
    assert by_id["b"].fusion_score == pytest.approx(1 / 12)
    assert traces["a"].vector_rank == 1
    assert traces["a"].raw_vector_score == 0.9
    assert result.diagnostics.duplicate_backend_hit_count == 1


def test_noop_reranker_keeps_fused_order_and_leaves_rerank_scores_missing() -> None:
    retriever, _, _, _ = _retriever(
        [_hit("b", "b", rank=1, vector=0.9), _hit("a", "a", rank=2, vector=0.8)]
    )

    result = retriever.retrieve("project-a", "query")

    assert isinstance(retriever.reranker, NoOpReranker)
    assert [hit.chunk.id for hit in result.hits] == ["b", "a"]
    assert all(hit.rerank_score is None for hit in result.hits)
    assert result.diagnostics.reranker_applied is False


def test_injected_reranker_receives_fused_candidates_and_can_change_order() -> None:
    reranker = DeterministicTestReranker({"a": 0.2, "b": 0.9})
    retriever, _, _, _ = _retriever(
        [_hit("a", "a", rank=1, vector=0.9), _hit("b", "b", rank=2, vector=0.8)],
        reranker=reranker,
    )

    result = retriever.retrieve("project-a", "  Keep CASE  ")

    assert reranker.queries == ["Keep CASE"]
    assert reranker.candidate_ids == ["a", "b"]
    assert [hit.chunk.id for hit in result.hits] == ["b", "a"]
    by_id = {hit.chunk.id: hit for hit in result.hits}
    assert by_id["b"].rerank_score == 0.9
    assert by_id["b"].vector_score == 0.8
    assert by_id["b"].fusion_score == pytest.approx(0.4)
    assert result.diagnostics.reranker_applied is True
    trace_scores = {
        trace.chunk_id: trace.rerank_score for trace in result.diagnostics.candidate_traces
    }
    assert trace_scores == {"a": 0.2, "b": 0.9}


def test_reranker_ties_use_fusion_score_then_chunk_id() -> None:
    reranker = DeterministicTestReranker({"z": 0.5, "a": 0.5, "b": 0.5})
    retriever, _, _, _ = _retriever(
        [
            _hit("z", "z", rank=1, vector=0.7),
            _hit("b", "b", rank=2, vector=0.8),
            _hit("a", "a", rank=3, vector=0.8),
        ],
        reranker=reranker,
    )

    result = retriever.retrieve("project-a", "query")

    assert [hit.chunk.id for hit in result.hits] == ["a", "b", "z"]


def test_substantially_redundant_overlapping_evidence_is_suppressed() -> None:
    short_text = "validate token and return a stable authentication result"
    long_text = short_text + " final"
    retriever, _, _, _ = _retriever(
        [
            _hit("long", long_text, rank=1, vector=0.95,
                 file_id="auth.py", start=1, end=10),
            _hit("short", short_text, rank=2, vector=0.8,
                 file_id="auth.py", start=2, end=9),
        ],
        [],
        overlap_line_threshold=0.8,
        overlap_text_threshold=0.8,
    )

    result = retriever.retrieve("project-a", "authentication", limit=5)

    assert [hit.chunk.id for hit in result.hits] == ["long"]
    assert result.diagnostics.overlap_suppression_count == 1
    trace = {item.chunk_id: item for item in result.diagnostics.candidate_traces}["short"]
    assert trace.selection_status == "suppressed_overlap"
    assert trace.selection_reason == "redundant_with:long"


def test_same_line_distinct_chunks_and_identical_text_in_other_files_survive() -> None:
    same_span_retriever, _, _, _ = _retriever(
        [
            _hit("auth", "validate_token checks signatures", rank=1, vector=0.9,
                 file_id="auth.py", start=5, end=5),
            _hit("roles", "authorize_roles reads account claims", rank=2, vector=0.8,
                 file_id="auth.py", start=5, end=5),
        ]
    )
    other_file_retriever, _, _, _ = _retriever(
        [
            _hit("one", "same evidence text", rank=1, vector=0.9, file_id="one.py"),
            _hit("two", "same evidence text", rank=2, vector=0.8, file_id="two.py"),
        ]
    )

    same_span = same_span_retriever.retrieve("project-a", "auth", limit=5)
    other_files = other_file_retriever.retrieve("project-a", "evidence", limit=5)

    assert {hit.chunk.id for hit in same_span.hits} == {"auth", "roles"}
    assert same_span.diagnostics.overlap_suppression_count == 0
    assert {hit.chunk.id for hit in other_files.hits} == {"one", "two"}
    assert other_files.diagnostics.overlap_suppression_count == 0


def test_diversity_defers_same_file_excess_then_backfills_and_ranks_contiguously() -> None:
    hits = [
        _hit(f"a{rank}", f"evidence {rank}", rank=rank, vector=1.0 - rank * 0.01,
             file_id="a.py")
        for rank in range(1, 5)
    ] + [
        _hit("b1", "other evidence", rank=5, vector=0.95, file_id="b.py"),
        _hit("c1", "more evidence", rank=6, vector=0.94, file_id="c.py"),
    ]
    retriever, _, _, _ = _retriever(hits, [], max_hits_per_file=3)

    result = retriever.retrieve("project-a", "evidence", limit=6)
    trace = {item.chunk_id: item for item in result.diagnostics.candidate_traces}

    assert [hit.chunk.id for hit in result.hits] == ["a1", "a2", "a3", "b1", "c1", "a4"]
    assert [hit.rank for hit in result.hits] == list(range(1, 7))
    assert trace["a4"].selection_status == "backfilled"
    assert trace["a4"].selection_reason == "backfilled_after_file_diversity"
    assert result.diagnostics.diversity_deferral_count == 1
    assert result.diagnostics.diversity_backfill_count == 1


def test_empty_channels_and_empty_results_are_valid() -> None:
    lexical_only, _, _, _ = _retriever([], [_hit("lex", "lexical", rank=1, lexical=2.0)])
    vector_only, _, _, _ = _retriever([_hit("vec", "vector", rank=1, vector=0.8)], [])
    empty, _, _, _ = _retriever()

    lexical_result = lexical_only.retrieve("project-a", "query")
    vector_result = vector_only.retrieve("project-a", "query")
    empty_result = empty.retrieve("project-a", "query")

    assert [hit.chunk.id for hit in lexical_result.hits] == ["lex"]
    assert lexical_result.hits[0].vector_score is None
    assert [hit.chunk.id for hit in vector_result.hits] == ["vec"]
    assert vector_result.hits[0].lexical_score is None
    assert empty_result.hits == ()
    assert empty_result.diagnostics.merged_candidate_count == 0


def test_foreign_project_hits_are_filtered_before_candidate_traces() -> None:
    retriever, _, _, _ = _retriever(
        [
            _hit("local", "local", rank=1, vector=0.8),
            _hit("foreign", "foreign", rank=2, vector=0.9, project_id="project-b"),
        ]
    )

    result = retriever.retrieve("project-a", "query")

    assert [hit.chunk.id for hit in result.hits] == ["local"]
    assert result.diagnostics.vector_candidate_count == 2
    assert result.diagnostics.filtered_foreign_project_count == 1
    assert all(trace.chunk_id != "foreign" for trace in result.diagnostics.candidate_traces)


def test_embedder_and_index_errors_propagate() -> None:
    retriever, embedder, vector, _ = _retriever()
    embedder.failure = RuntimeError("embedding model unavailable")
    with pytest.raises(RuntimeError, match="embedding model unavailable"):
        retriever.retrieve("project-a", "query")

    embedder.failure = None
    vector.failure = RuntimeError("vector index damaged")
    with pytest.raises(RuntimeError, match="vector index damaged"):
        retriever.retrieve("project-a", "query")


def test_fusion_configuration_rejects_invalid_weights_and_strategy() -> None:
    embedder = FixedEmbedder()
    vector = FixedVectorIndex()
    lexical = FixedLexicalIndex()

    with pytest.raises(ValueError, match="weights must be non-negative"):
        HybridRetriever(
            embedder=embedder,
            vector_index=vector,
            lexical_index=lexical,
            vector_weight=-1.0,
        )
    with pytest.raises(ValueError, match="sum of fusion weights must be greater than zero"):
        HybridRetriever(
            embedder=embedder,
            vector_index=vector,
            lexical_index=lexical,
            vector_weight=0.0,
            lexical_weight=0.0,
        )
    with pytest.raises(ValueError, match="unsupported fusion strategy"):
        HybridRetriever(
            embedder=embedder,
            vector_index=vector,
            lexical_index=lexical,
            fusion_strategy="unknown",
        )
