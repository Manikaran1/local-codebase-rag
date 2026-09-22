# Retrieval Pipeline

## Objective

Return a small, high-quality evidence set for a repository question.

## Query path

```text
question
  ↓
query preprocessing
  ↓
┌───────────────────────┐
│ vector candidate set  │
└──────────┬────────────┘
           │
           ├─────────────┐
           │             │
           ▼             ▼
      semantic        lexical/BM25
      retrieval       retrieval
           │             │
           └──────┬──────┘
                  ▼
             deduplicate
                  ▼
              score fuse
                  ▼
              reranking
                  ▼
          evidence selection
                  ▼
           context builder
                  ▼
               LLM
```

## Candidate retrieval

Vector search is expected to perform well on conceptual questions:
- "Where is session validation implemented?"
- "How does retry behavior work?"

Lexical search is expected to rescue exact terms:
- function names
- error strings
- configuration keys
- class names
- constants
- API routes

Neither search should be treated as universally superior.

## Fusion

The first implementation should expose the raw component scores and fused score.

A retrieval debug command should eventually support output similar to:

```text
#1 src/auth/service.py:41-88
vector: 0.82
bm25:   0.67
fusion: 0.77

#2 src/api/login.py:17-39
vector: 0.71
bm25:   0.83
fusion: 0.76
```

This makes failures diagnosable.

## Reranking

Reranking is optional in the earliest implementation but the interface should exist.

Candidate strategies:
- local cross-encoder
- lightweight local reranker
- heuristic structural boost
- no reranker baseline

The project should benchmark whether a reranker materially improves retrieval before accepting its latency cost.

## Structural boosts

Potential bounded ranking features:
- exact symbol-name match
- path-name match
- symbol kind
- query mentions identifier
- same-file relationship
- definition over reference preference for "where is X defined?"
- test file downweight for implementation questions

These must remain measurable ranking features, not opaque hand-written answer logic.

## Evidence selection

The final evidence set should avoid:
- duplicate overlapping chunks
- ten chunks from one file when three files provide complementary evidence
- generated summaries with no source span
- excessively large chunks

## Failure behavior

If retrieval evidence is weak, generation should say so.

The system should not convert low-confidence retrieval into a high-confidence hallucination.
