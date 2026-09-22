# ADR-0002: Hybrid retrieval

**Status:** Accepted

## Context

Semantic embeddings work well for conceptual similarity but can miss exact identifiers, configuration keys, error strings, and uncommon names. Lexical retrieval has the opposite tradeoff.

## Decision

v0.1 will maintain both vector and lexical indexes and combine their candidate results.

## Consequences

- retrieval is more robust across question types
- indexing complexity increases
- score fusion must be measurable and evaluated
- storage requirements increase modestly
