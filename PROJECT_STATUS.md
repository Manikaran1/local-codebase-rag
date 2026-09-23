# Project Status

## Current state

**Stage:** Phase 1 complete; next execution unit is combined Phase 2–3 parsing, document ingestion, and chunking (Issue #2)
**Target:** v0.1.0  
**Visibility recommendation:** Private until the retrieval core is demonstrably usable.

## Completed foundation

- [x] v0.1 scope
- [x] high-level architecture
- [x] module boundaries
- [x] domain contracts
- [x] local-first ADR
- [x] hybrid retrieval ADR
- [x] code-aware chunking ADR
- [x] package skeleton
- [x] CI definition
- [x] test fixture skeleton
- [x] evaluation strategy
- [x] roadmap
- [x] security guidance
- [x] deterministic repository scanner and supported-file filtering
- [x] .gitignore handling and default exclusions
- [x] binary and maximum-size protection
- [x] stable source IDs and content hashing

## Next implementation tasks

- [ ] Issue #2 — implement combined parsing, Markdown/text ingestion, Python symbol extraction, and structural/prose chunking
- [ ] Issue #4 — implement local embeddings, vector index, and BM25 index
- [ ] Issue #5 — implement hybrid retrieval, fusion, reranking, and diagnostics
- [ ] Issue #6 — implement citation-aware context building, local generation, and CLI workflow
- [ ] Issue #7 — build retrieval evaluation and regression gates
- [ ] Issue #8 — expose stable application services through FastAPI

## Planning note

GitHub Issue #3 has been folded into Issue #2. Parsing and chunking now form one implementation unit so symbol/document boundaries, source spans, chunk metadata, and deterministic IDs are designed and tested together rather than across two partially overlapping phases.
