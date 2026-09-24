# Retrieval Pipeline

## Objective

Return a small, project-scoped evidence set for a repository question and keep the ranking
inspectable without calling a language model.

## Query and candidate path

```text
question
   ↓
collapse whitespace, preserve case and punctuation
   ├───────────────────────┐
   ↓                       ↓
embed once             lexical query
   ↓                       ↓
VectorIndex            LexicalIndex (BM25)
   └───────────┬───────────┘
               ↓
       merge by chunk ID
               ↓
       weighted fusion or RRF
               ↓
        optional reranker
               ↓
    redundant evidence control
               ↓
       per-file diversity
               ↓
     ranked SearchHit tuple
      + retrieval diagnostics
```

`HybridRetriever.retrieve(project_id, query, limit=10)` passes the project ID to both index
interfaces. It never reads SQLite or relies on backend-specific methods. The normalized query is
`" ".join(query.split())`; identifiers, letter case, and punctuation are left intact. Blank
queries raise `ValueError`. The normalized query is embedded exactly once and sent unchanged to the
lexical index.

Both indexes receive `max(limit, retrieval_candidate_limit)` as their candidate limit. The default
configured candidate limit is 40, so a caller asking for more than 40 final results is not silently
under-served. Each index returns its own rank and raw score. Retrieval keeps vector and lexical
ranks separately while merging on `chunk.id`.

## Fusion and score meanings

The default strategy is normalized weighted fusion. For a query's candidate set:

```text
vector_normalized = min(1, max(0, raw_cosine_similarity))
max_bm25 = max(raw BM25 scores, default=0)
lexical_normalized = raw_bm25 / max_bm25, if max_bm25 > 0; otherwise 0
wv = vector_weight / (vector_weight + lexical_weight)
wl = lexical_weight / (vector_weight + lexical_weight)
fusion_score = wv * vector_normalized + wl * lexical_normalized
```

The default weights are 0.5 and 0.5. Weights must be finite, non-negative, and have a positive
sum. Missing channel scores remain `None` in `SearchHit` and contribute zero to fusion. Raw
vector and BM25 values are never overwritten by normalized values.

The optional `rrf` strategy uses each channel's independent one-based rank:

```text
fusion_score = vector_weight / (rrf_k + vector_rank)
             + lexical_weight / (rrf_k + lexical_rank)
```

A missing channel contributes nothing. `rrf_k` defaults to 60 and must be positive. The configured
weights are used directly in RRF; weighted fusion normalizes their sum as shown above. Both
strategies sort equal scores by `chunk_id` ascending.

`fusion_score` is a ranking score, not a calibrated probability or answer-confidence value. No
confidence threshold is applied. Vector search continues to expose raw cosine similarity and BM25
continues to expose its raw score; normalization and fusion start in this retrieval layer.

## Reranking

`Reranker` accepts the normalized query and fused candidates with their raw scores, independent
channel ranks, and component normalizations. An active reranker returns one finite score per
candidate chunk ID. Results then sort by rerank score descending, fusion score descending, and
chunk ID ascending. Raw component scores and fusion scores remain available in final hits and
diagnostics.

The default is `NoOpReranker`: it makes no model call, adds no rerank score, and preserves fused
order. Test rerankers can implement the same narrow interface. No cross-encoder, model, or remote
reranking service is part of this phase.

## Redundant evidence and source diversity

Different chunk IDs are not suppressed from line overlap alone. A lower-ranked chunk is suppressed
only when it is from the same file, its line-span overlap coefficient is at least the configured
line threshold, and its whitespace-normalized text is contained in the other text with a length
ratio at least the configured text threshold. Both thresholds default to 0.8. This containment
rule keeps different text fragments with the same line span eligible and does not collapse similar
text from different files.

After redundancy suppression, selection makes a first pass through ranked candidates and takes at
most three chunks per file by default. Further candidates from a capped file are deferred. If that
first pass returns fewer than the requested limit, the best deferred candidates are appended as a
second-pass backfill. Diversity does not change relevance scores, and final `SearchHit.rank` values
are contiguous from 1.

## Results and diagnostics

`RetrievalResult` contains an immutable tuple of final `SearchHit` values and a separate
`RetrievalDiagnostics` value. `SearchHit.rank` is only the final result rank; vector and lexical
ranks stay in the retrieval candidate trace. Each trace records:

- chunk ID and each available channel rank;
- raw vector and lexical scores;
- normalized vector and lexical contributions;
- fusion and optional rerank scores;
- final rank, if selected;
- selection state and reason, including overlap suppression, diversity deferral, and backfill.

Query diagnostics also include the normalized query, project ID, strategy name, vector, lexical,
and merged candidate counts, configured and effective candidate limits, final limit, whether
reranking ran, overlap suppression count, diversity deferral and backfill counts, duplicate
backend-hit count, and foreign-project hits filtered from malformed backend responses. Diagnostics
are programmatic data; this phase does not add a terminal debug command.

A valid query with no indexed candidates returns an empty result with diagnostics. Either channel
may return no candidates. Embedding and index errors propagate to the caller; retrieval does not
silently switch to lexical-only behavior. Query and repository content remain local.

## Focused regression checks

The sample repository is indexed through the scanner, source processor, `IndexingService`, and the
SQLite vector and BM25 implementations. A controlled, deterministic test embedder exercises
authentication, database, retry/configuration, documentation, and cross-file queries without a
download or an LLM. These are pipeline-mechanics checks rather than a claim of production model
quality. The initial gates require single-source evidence within top 5 and all required cross-file
sources within top 10; broader retrieval quality evaluation remains in Phase 7.
