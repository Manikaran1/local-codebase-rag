# Architecture

## 1. Purpose

Local Codebase RAG is a modular local retrieval system for software repositories and technical documentation.

The v0.1 architecture is intentionally separated into ingestion, parsing, chunking, indexing, retrieval, context construction, and generation. No layer should need to know the implementation details of a non-adjacent layer.

## 2. System boundary

### Inputs
- local repository path
- repository configuration
- source files
- Markdown / text documentation
- user question

### Outputs
- indexed project representation
- ranked evidence
- grounded answer
- source citations
- diagnostic retrieval metadata

### External runtime
- optional local Ollama-compatible inference server

No cloud service is required by the architecture.

## 3. Primary pipeline

```text
Repository
   │
   ▼
RepositoryScanner
   │
   ├── ignore handling
   ├── size limits
   ├── extension/language detection
   └── binary exclusion
   │
   ▼
SourceDocument
   │
   ├───────────────┐
   ▼               ▼
CodeParser     DocumentParser
   │               │
   ▼               ▼
ParsedSymbol    ParsedDocument
   │               │
   ├───────┬───────┘
           ▼
        Chunker
           │
           ▼
     IndexedChunk
           │
      ┌────┴────┐
      ▼         ▼
 Embeddings   BM25
      │         │
      └────┬────┘
           ▼
    HybridRetriever
           │
           ▼
       Reranker
           │
           ▼
     ContextBuilder
           │
           ▼
       LocalLLM
           │
           ▼
 Answer + Citation[]
```

## 4. Layer responsibilities

### `domain`
Owns shared data contracts. Other modules exchange domain objects instead of implementation-specific dictionaries.

Core contracts:
- `SourceFile`
- `SourceSpan`
- `Symbol`
- `Chunk`
- `SearchHit`
- `Citation`
- `Answer`

### `config`
Loads application configuration and validates runtime settings.

Configuration should cover:
- project storage location
- file limits
- ignored globs
- embedding backend/model
- vector backend
- retrieval weights
- candidate limits, weighted/RRF strategy, and per-file selection thresholds
- reranking
- local LLM endpoint/model
- context token budget

### `ingestion`
Discovers candidate files.

Responsibilities:
- repository traversal
- `.gitignore` awareness
- default ignore list
- binary detection
- maximum file size
- supported extensions
- stable file identifiers

It must not parse syntax or create embeddings.

### `parsing`
Transforms source files into structural metadata.

For code, parsing should eventually expose:
- language
- imports
- symbol names
- symbol kinds
- start/end lines
- parent symbol
- signatures
- documentation/comments where useful

The parser interface must support multiple language implementations.

### `chunking`
Creates retrieval units.

Chunking policy:
1. prefer complete symbols when under size limits;
2. preserve a symbol's metadata;
3. split oversized symbols using bounded secondary rules;
4. chunk prose using headings/paragraph boundaries;
5. preserve source line ranges;
6. generate stable chunk IDs from source identity + content.

### `embeddings`
Converts textual chunk representations into vectors.

This layer should not know which vector store is used. The initial Sentence Transformers adapter is
lazy, CPU-compatible, and local-files-only unless model acquisition is explicitly enabled.

### `index`
Owns persistence and search primitives.

Two independent capabilities are required:
- vector similarity
- lexical/BM25

The v0.1 reference backend stores chunks, float32 vectors, lexical tokens, and project index
metadata in one SQLite database. Vector search uses NumPy exact cosine similarity. Searches are
project-scoped, and file replacement updates both indexes in one SQLite transaction. Raw vector and
lexical scores remain independent; fusion belongs to `retrieval`.

### `retrieval`
Combines candidate retrieval and ranking.

Expected steps:
1. normalize query
2. vector candidate retrieval
3. BM25 candidate retrieval
4. merge/deduplicate
5. score fusion
6. optional rerank
7. conservative same-file redundant-evidence suppression
8. per-file diversity selection with deferred-candidate backfill
9. return final-ranked evidence with retrieval-owned diagnostics

The retriever depends on `Embedder`, `VectorIndex`, and `LexicalIndex` interfaces. It preserves
raw vector and BM25 scores, keeps each channel's rank independent, and assigns `SearchHit.rank`
only after final selection. Weighted fusion normalizes cosine and query-local BM25 contributions;
RRF is an alternative based on channel ranks. Neither score is an answer-confidence estimate.
Diagnostics belong to retrieval and do not alter chunk metadata.

### `generation`
Turns evidence into a bounded prompt and generates a response.

Rules:
- generation must receive explicit retrieved chunks;
- citations are derived from evidence metadata, not invented by the LLM;
- context must fit a configurable budget;
- system instructions must require uncertainty when evidence is insufficient.

### `cli`
Primary v0.1 product interface.

Commands:
- `index`
- `ask`
- later `status`, `remove`, `inspect`

### `api`
Thin wrapper around application services after the CLI path is stable.

The API must not duplicate indexing/retrieval logic.

## 5. Domain model

### SourceFile

```text
id
project_id
relative_path
absolute_path
language
content_hash
size_bytes
```

### SourceSpan

```text
start_line
end_line
start_byte?
end_byte?
```

### Symbol

```text
id
file_id
name
qualified_name
kind
language
span
parent_id?
signature?
```

### Chunk

```text
id
project_id
file_id
relative_path
language
text
span
symbol_id?
symbol_name?
symbol_kind?
content_hash
metadata
```

### SearchHit

```text
chunk
vector_score?
lexical_score?
fusion_score?
rerank_score?
rank
```

### Citation

```text
relative_path
start_line
end_line
symbol_name?
chunk_id
```

## 6. Stable identity

Index correctness depends on deterministic IDs.

Suggested scheme:

```text
project_id = hash(canonical_repository_path)
file_id    = hash(project_id + relative_path)
chunk_id   = hash(file_id + span + content_hash + source_offset)
```

`source_offset` is a zero-based character offset used to distinguish identical chunks that share
the same line-only span. Descriptive metadata does not participate in chunk identity.

Repository moves may later require user-defined project IDs.

## 7. Incremental indexing

v0.1 may begin with complete re-indexing for correctness.

The architecture must nevertheless preserve the metadata needed for incremental indexing:

```text
file path
content hash
parser version
chunker version
embedding provider, model, immutable revision/fingerprint, dimension, and normalization policy
```

The index stores this embedding identity with each project's vector metadata and rejects a search or
upsert when the requested identity differs. Existing schema versions without revision metadata must
be backed up and re-indexed before use with the current index schema.

A later indexer can then classify each source as:
- unchanged
- modified
- added
- deleted

## 8. Retrieval fusion

Initial fusion can use normalized weighted scoring:

```text
fusion =
    vector_weight  * normalized_vector_score
  + lexical_weight * normalized_bm25_score
```

Reciprocal Rank Fusion is another candidate and should be evaluated rather than chosen by intuition.

Weights must be configuration, not constants embedded in business logic.

## 9. Context construction

The context builder receives ranked hits and should:

1. deduplicate overlapping spans;
2. preserve source metadata;
3. prevent one large file from consuming the full prompt;
4. respect a token/character budget;
5. order context predictably;
6. produce citation handles that map directly to evidence.

## 10. Trust boundary

Local source code is potentially sensitive.

The default architecture therefore prohibits silent network transmission of:
- repository files
- chunks
- embeddings
- prompts
- generated repository summaries

Any future remote provider must be explicit, opt-in, and isolated behind a provider adapter.

## 11. Extension points

The following are formal extension points:
- parser
- chunker
- embedding provider
- vector store
- lexical index
- fusion strategy
- reranker
- generation provider

They should not become a general plugin framework in v0.1.

## 12. Dependency direction

```text
domain
 ↑
config

domain ← ingestion
domain ← parsing
domain ← chunking
domain ← embeddings
domain ← index

retrieval → domain + index + embeddings
generation → domain + retrieval
cli → application services
api → application services
```

Infrastructure implementations may depend on domain interfaces. Domain objects must never depend on specific infrastructure packages.

## 13. Architectural rules

1. No direct Ollama calls outside `generation`.
2. No direct vector database calls outside `index`.
3. No parser-specific Tree-sitter objects outside `parsing`.
4. No UI/API logic in retrieval.
5. No generated citation accepted without matching evidence metadata.
6. No remote network dependency in the core indexing path.
7. No arbitrary global mutable state.
8. Configuration must be injectable for tests.
9. Retrieval scores must be inspectable.
10. Tests must be able to substitute deterministic fake embeddings and generators.
