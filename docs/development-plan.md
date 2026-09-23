# v0.1 Development Plan

The implementation order is designed so every stage produces a testable system boundary. GitHub issues are the execution units; this document defines the architectural scope and exit conditions for those issues.

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

## Phase 1 — Repository ingestion — GitHub Issue #1

Deliverables:
- repository scanner
- file filtering
- ignore handling
- binary rejection
- file-size protection
- content hashing

Exit condition:
- fixture repository produces the expected accepted file manifest

Status:
- complete and merged

## Phase 2–3 — Parsing, document ingestion, and chunking — GitHub Issue #2

Phases 2 and 3 are implemented as one execution unit because parsing and chunking share the same source-span, symbol, metadata, and document-boundary contracts. The former GitHub Issue #3 is folded into Issue #2 and should not be implemented separately.

Deliverables:
- parser interface
- plain-text / Markdown document parsing path
- Tree-sitter adapter boundary
- first language implementation (Python)
- Python function/class symbol extraction
- symbol name, kind, qualified name, signature, parent, and source span metadata
- safe fallback behavior for unsupported or malformed source
- structural code chunker
- prose / Markdown chunker
- symbol-boundary preference
- oversized-symbol fallback splitting
- stable chunk IDs and content hashes
- file identity, source span, symbol metadata, and document metadata preservation

Exit condition:
- known fixture Python classes/functions are extracted with correct line spans
- parser-specific objects do not escape the parsing module
- malformed or unsupported inputs fail safely
- Markdown/text inputs produce deterministic prose chunks that respect headings and paragraph boundaries where possible
- complete code symbols remain intact when under configured chunk limits
- oversized symbols split deterministically
- every chunk retains its source file identity and source span
- chunk IDs/content hashes are deterministic

## Phase 4 — Indexing — GitHub Issue #4

Deliverables:
- embedding interface
- local sentence-transformer implementation
- vector index interface + first backend
- BM25 index
- project/file delete and re-index behavior
- persistence layout and version metadata

Exit condition:
- chunks can be indexed, retrieved, removed, and re-indexed deterministically

## Phase 5 — Hybrid retrieval — GitHub Issue #5

Deliverables:
- vector retriever
- lexical retriever
- result deduplication
- score normalization
- configurable fusion
- retrieval diagnostics
- optional reranker contract
- source diversity handling

Exit condition:
- evaluation questions retrieve required evidence in top-k at agreed thresholds
- raw vector, lexical, and fused scores remain inspectable

## Phase 6 — Citation-aware generation and CLI — GitHub Issue #6

Deliverables:
- bounded context builder
- overlap/dedup handling
- citation mapping
- local LLM interface
- Ollama-compatible implementation
- evidence-constrained generation prompt
- `codebase-rag index`
- `codebase-rag ask`
- readable errors and retrieval debug output

Exit condition:
- answer citations always point to retrieved chunks
- weak evidence produces explicit uncertainty rather than fabricated certainty
- fresh repository → index → question → grounded answer + citations works from the terminal

## Phase 7 — Evaluation hardening — GitHub Issue #7

Deliverables:
- query/evidence gold set
- Hit@K / Recall@K
- MRR where appropriate
- required-source coverage
- duplicate-evidence rate
- regression comparison tooling
- latency measurements
- semantic, identifier, cross-file, and documentation query groups

Exit condition:
- benchmark runs locally and in CI without an LLM
- retrieval changes cannot silently degrade agreed quality thresholds
- results are reproducible and documented

## Phase 8 — API — GitHub Issue #8

Deliverables:
- FastAPI application
- project index endpoint
- query endpoint
- status endpoint
- project deletion endpoint
- request/response schemas
- reuse of existing application services

Exit condition:
- API reuses application services and does not duplicate core logic
- local-only defaults remain preserved
- integration tests exercise the index/query lifecycle

## Definition of v0.1.0

v0.1.0 requires:
- local repository indexing
- at least one structurally parsed programming language
- Markdown/text ingestion
- structural code and prose chunking
- vector + lexical retrieval
- hybrid ranking
- local generation
- citations
- CLI
- automated evaluation
- reproducible setup
