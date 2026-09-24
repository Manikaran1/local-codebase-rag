# Local Codebase RAG

A **local-first retrieval-augmented generation assistant for understanding software repositories and technical documentation**.

Local Codebase RAG is designed to index a software project, understand source structure, retrieve relevant code and documentation through hybrid search, and answer questions with precise file and line-level citations — while keeping source code and project context on the local machine.

> **Status:** Private early-development repository<br>
> **Current target:** v0.1.0 — reliable local codebase indexing, hybrid retrieval, local generation, and source-grounded answers.

---

## Why this project exists

General-purpose "chat with your files" systems often treat source code as plain text. That loses important structure: classes, functions, imports, symbol relationships, file hierarchy, and source locations.

This project takes a code-aware approach.

Instead of only asking:

> "Which text chunks are semantically similar to the question?"

the system is intended to ask:

> "Which source symbols, documentation passages, files, lexical matches, and structural relationships provide the strongest evidence for the answer?"

The initial focus is **retrieval quality, traceability, and local execution** — not autonomous code editing.

---

## v0.1 Goals

The first usable release should be able to:

1. Index a local software repository.
2. Respect ignore rules and configurable file filters.
3. Parse supported programming languages with code-aware parsers.
4. Extract symbols and source metadata.
5. Chunk code by structure rather than arbitrary token windows where possible.
6. Ingest Markdown, text, and later PDF documentation.
7. Produce local embeddings.
8. Maintain both vector and lexical indexes.
9. Perform hybrid retrieval.
10. Rerank candidate context.
11. Build a bounded prompt context.
12. Query a local LLM.
13. Return an answer with file, symbol, and line citations.
14. Expose the workflow through a CLI.
15. Provide a small API after the CLI path is stable.
16. Include reproducible tests and a retrieval evaluation corpus.

---

## Non-goals for v0.1

The following are deliberately out of scope for the first release:

- autonomous code modification
- multi-agent workflows
- IDE plugins
- cloud model APIs as a requirement
- multi-user accounts
- remote project synchronization
- automatic pull requests
- repository-wide refactoring
- long-term conversational memory
- training or fine-tuning a custom LLM
- complex web UI

These can be evaluated after the retrieval core is trustworthy.

---

## High-level architecture

```text
Local Repository / Documentation
            │
            ▼
      Repository Scanner
            │
            ▼
     Parsing + Metadata
            │
      ┌─────┴─────┐
      ▼           ▼
 Code Chunking  Text Chunking
      │           │
      └─────┬─────┘
            ▼
      Canonical Documents
            │
      ┌─────┴──────────┐
      ▼                ▼
 Embedding Index   Lexical Index
   (Vector)           (BM25)
      │                │
      └───────┬────────┘
              ▼
       Hybrid Retrieval
              │
              ▼
           Reranker
              │
              ▼
        Context Builder
              │
              ▼
          Local LLM
              │
              ▼
     Answer + Citations
```

For a deeper breakdown, see [`docs/architecture.md`](docs/architecture.md).

---

## Planned technology stack

| Area | Initial choice |
|---|---|
| Language | Python 3.11+ |
| CLI | Typer |
| API | FastAPI |
| Validation/config | Pydantic + pydantic-settings |
| Parsing | Tree-sitter + language adapters |
| Embeddings | Sentence Transformers |
| Vector storage | SQLite + NumPy exact cosine reference backend |
| Lexical retrieval | BM25 |
| Local generation | Ollama-compatible adapter |
| Testing | pytest |
| Linting | Ruff |
| Type checking | mypy |
| CI | GitHub Actions |

Storage implementations are intentionally behind interfaces so a backend can be replaced without changing retrieval logic.

---

## Repository layout

```text
local-codebase-rag/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/
│   └── PULL_REQUEST_TEMPLATE.md
├── docs/
│   ├── adr/
│   ├── architecture.md
│   ├── development-plan.md
│   ├── evaluation.md
│   ├── installation.md
│   ├── retrieval-pipeline.md
│   └── roadmap.md
├── examples/
│   └── sample_repo/
├── scripts/
├── src/
│   └── codebase_rag/
│       ├── api/
│       ├── chunking/
│       ├── cli/
│       ├── config/
│       ├── domain/
│       ├── embeddings/
│       ├── generation/
│       ├── index/
│       ├── ingestion/
│       ├── parsing/
│       └── retrieval/
├── tests/
│   ├── integration/
│   └── unit/
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── README.md
└── SECURITY.md
```

---

## Intended CLI

The first user-facing workflow is planned around two commands:

```bash
codebase-rag index ./path/to/project
```

and:

```bash
codebase-rag ask "Where is authentication handled?"
```

Expected answer shape:

```text
Authentication is coordinated by AuthService.authenticate() and used by
the login endpoint before a session is created.

Sources:
- src/auth/service.py:41-88 — AuthService.authenticate
- src/api/login.py:17-39 — login
```

The answer should be treated as incomplete when the retrieval evidence is weak rather than allowing the model to confidently invent repository behavior.

---

## Core design principles

### Local-first

Source code, indexes, prompts, and model interaction should remain local by default.

### Evidence before generation

Generation is downstream of retrieval. The assistant should expose the sources used to construct an answer.

### Code-aware indexing

Code is not generic prose. Structural parsing should preserve symbols, line ranges, parent/child relationships, language, and file metadata.

### Hybrid retrieval

Semantic similarity and exact lexical matching solve different retrieval problems. v0.1 will combine both instead of depending on only one.

### Bounded interfaces

Parsers, embedding models, index backends, rerankers, and local LLM runtimes should be replaceable through explicit interfaces.

### Evaluation-driven development

Retrieval quality must be measured against a known test repository rather than judged only from impressive-looking demos.

---

## Current development state

The project currently contains:

- architecture definition
- v0.1 scope
- domain contracts
- module boundaries
- project configuration
- implemented repository ingestion and filtering
- source-package skeleton for later stages
- test and evaluation foundations
- CI definition
- contribution and security guidance

### Repository discovery

Phase 1 provides RepositoryScanner.scan(path) as the local ingestion boundary. It returns shared
SourceFile metadata in stable relative-path order, applies repository .gitignore files and a small
built-in exclusion list, and accepts supported source and documentation extensions. Files larger
than the configurable CODEBASE_RAG_MAX_FILE_SIZE_BYTES limit (2 MiB by default), binary files,
unsupported extensions, and symbolic links beneath the repository root are skipped. A symlink
passed as the repository root is resolved to its target before scanning. Scanning performs no
network calls and does not parse file contents.

The supported-extension map and scanner exclusions are defined in Settings and can be overridden
when constructing a scanner.

### Parsing and chunking

Issue #2 adds a local source-to-chunk path through `SourceProcessor`. Python files use Tree-sitter
to extract classes, functions, methods, signatures, parent links, and one-based source spans.
Markdown files are divided into headings, paragraphs, and fenced code blocks; plain-text files are
divided into paragraphs. Other scanned languages use an explicit plain-text fallback until a
structural parser is added. Tree-sitter nodes stay inside the Python parsing adapter.
If Tree-sitter reports syntax errors, definition subtrees containing errors are omitted and a
diagnostic is returned; valid sibling definitions remain available. The parser does not invent
symbols for malformed input.

Code chunks keep complete symbols when they fit the configurable
`CODEBASE_RAG_MAX_CHUNK_CHARS` limit (4,000 characters by default). Oversized symbols split on
nested symbol and line boundaries where possible, then at a fixed character boundary. Prose
chunks prefer document blocks and sentence or line boundaries. Chunk IDs and SHA-256 content hashes
are deterministic, and every chunk carries file identity, source line spans, and applicable symbol
or heading metadata.

### Local indexing

Issue #4 adds independent project-scoped vector and BM25 indexes backed by
`.codebase-rag/indexes/index.sqlite3`. The SQLite reference backend stores complete chunk metadata
and float32 vectors; NumPy computes exact cosine scores. BM25 uses a deterministic tokenizer that
preserves full identifiers and adds case-folded snake_case, camelCase, and PascalCase parts. File
replacement updates both indexes in one transaction, and project/file deletion removes their
persisted records.

NumPy is installed with the core package. Install the optional Sentence Transformers provider with
`pip install -e ".[dev,local-embeddings]"`. The default model is
`sentence-transformers/all-MiniLM-L6-v2`, pinned to commit
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; it loads lazily from a local path or cache. For another
Hub model, set `CODEBASE_RAG_EMBEDDING_REVISION` to its immutable 40-character commit SHA. For a
local model directory, set it to a stable version or content fingerprint and update that identity
whenever the model files change. Model downloads are disabled by default and require the explicit
`CODEBASE_RAG_EMBEDDING_ALLOW_DOWNLOAD=true` setting. Repository chunks are never sent to an
embedding API. Vector and BM25 searches expose raw component scores independently; score fusion and
reranking remain later retrieval work.

---

## Development order

GitHub issues are the execution units for v0.1. Parsing, document ingestion, and chunking are intentionally implemented together because they share source-boundary and metadata contracts.

```text
1. Repository ingestion and filtering — Issue #1 (complete)
2. Parsing + document ingestion + structural/prose chunking — Issue #2 (complete and merged)
   - Issue #3 is folded into Issue #2 and is not implemented separately
3. Local embeddings + vector index + BM25 — Issue #4
4. Hybrid retrieval + fusion + reranking — Issue #5
5. Citation-aware generation + CLI workflow — Issue #6
6. Retrieval evaluation and regression gates — Issue #7
7. FastAPI application surface — Issue #8
8. Additional language support — after the core v0.1 path is stable
```

See [`docs/development-plan.md`](docs/development-plan.md).

---

## Release target: v0.1.0

A v0.1.0 release should only be created when a fresh user can:

```text
repository
    ↓
index
    ↓
ask a repository question
    ↓
receive a grounded answer
    ↓
inspect correct source citations
```

and the same workflow passes a deterministic retrieval evaluation suite.

---

## Privacy

This project is specifically intended for codebases that users may not want sent to remote AI services. The architecture therefore assumes local processing as the default security boundary.

See [`SECURITY.md`](SECURITY.md).

---

## License

The project is currently scaffolded with the Apache License 2.0 in preparation for a future public release.

Before the repository is made public, dependencies, model licenses, parser grammars, sample data, and bundled assets must be reviewed for redistribution compatibility.
