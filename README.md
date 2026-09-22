# Local Codebase RAG

A **local-first retrieval-augmented generation assistant for understanding software repositories and technical documentation**.

Local Codebase RAG is designed to index a software project, understand source structure, retrieve relevant code and documentation through hybrid search, and answer questions with precise file and line-level citations — while keeping source code and project context on the local machine.

> **Status:** Private pre-development repository  
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
| Vector storage | Adapter-based; initial local backend TBD |
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
- source-package skeleton
- test skeleton
- evaluation plan
- CI definition
- contribution and security guidance

Most source modules are intentionally interfaces or stubs at this stage. Implementation should proceed from the ingestion and domain layers outward.

---

## Development order

```text
1. Domain models and configuration
2. Repository scanner
3. Text ingestion
4. Code parser abstraction
5. Python / initial language parser
6. Structural chunking
7. Embedding adapter
8. Vector index
9. BM25 lexical index
10. Hybrid retrieval
11. Reranking
12. Citation-aware context builder
13. Local LLM adapter
14. CLI
15. Retrieval evaluation suite
16. FastAPI service
17. Additional language support
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
