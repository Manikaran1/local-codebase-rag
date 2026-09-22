# Project Status

## Current state

**Stage:** Phase 1 — repository ingestion complete
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

- [ ] implement text ingestion
- [ ] introduce Tree-sitter parser adapter
- [ ] implement Python symbol parser
- [ ] implement structural chunker
- [ ] choose first vector backend based on benchmark/operational fit
- [ ] implement embedding provider
- [ ] implement BM25 index
- [ ] implement hybrid retrieval baseline
