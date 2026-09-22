# v0.1 Development Plan

The implementation order is designed so every stage produces a testable system boundary.

## Phase 0 — Foundation

Deliverables:
- repository structure
- domain models
- configuration
- logging conventions
- CI
- deterministic fixture repository
- architecture docs

Exit condition:
- package installs
- static checks pass
- tests run

## Phase 1 — Repository ingestion

Deliverables:
- repository scanner
- file filtering
- ignore handling
- binary rejection
- file-size protection
- content hashing

Exit condition:
- fixture repository produces the expected accepted file manifest

## Phase 2 — Parsing

Deliverables:
- parser interface
- plain-text parser
- Tree-sitter adapter
- first language implementation (Python)
- symbol metadata

Exit condition:
- known fixture classes/functions are extracted with correct line spans

## Phase 3 — Chunking

Deliverables:
- structural code chunker
- prose chunker
- stable chunk IDs
- oversized-symbol splitting
- metadata preservation

Exit condition:
- chunks never lose their source file and line span

## Phase 4 — Indexing

Deliverables:
- embedding interface
- local sentence-transformer implementation
- vector index interface + first backend
- BM25 index
- project lifecycle operations

Exit condition:
- chunks can be indexed, retrieved, removed, and re-indexed deterministically

## Phase 5 — Hybrid retrieval

Deliverables:
- vector retriever
- lexical retriever
- result deduplication
- fusion
- retrieval diagnostics
- optional reranker contract

Exit condition:
- evaluation questions retrieve required evidence in top-k at agreed thresholds

## Phase 6 — Generation

Deliverables:
- context builder
- citation mapping
- local LLM interface
- Ollama-compatible implementation
- evidence-constrained generation prompt

Exit condition:
- answer citations always point to retrieved chunks

## Phase 7 — CLI

Deliverables:
- `codebase-rag index`
- `codebase-rag ask`
- readable errors
- debug retrieval output

Exit condition:
- fresh repository → index → question → grounded answer works from terminal

## Phase 8 — Evaluation hardening

Deliverables:
- query/evidence gold set
- hit-rate metrics
- MRR / recall@k where appropriate
- regression tests
- latency measurements

Exit condition:
- retrieval changes cannot silently degrade the benchmark

## Phase 9 — API

Deliverables:
- FastAPI application
- project index endpoint
- query endpoint
- status endpoint
- project deletion endpoint

Exit condition:
- API reuses application services and does not duplicate core logic

## Definition of v0.1.0

v0.1.0 requires:
- local repository indexing
- at least one structurally parsed programming language
- Markdown/text ingestion
- vector + lexical retrieval
- hybrid ranking
- local generation
- citations
- CLI
- automated evaluation
- reproducible setup
