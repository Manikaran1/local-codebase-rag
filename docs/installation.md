# Installation (Development)

> The project is currently a scaffold; dependency choices may change before v0.1.

## Requirements

- Python 3.11+
- Git
- optional: Ollama-compatible local inference runtime

## Development install

```bash
git clone <repository-url>
cd local-codebase-rag

python -m venv .venv
```

Activate the environment, then:

```bash
pip install -e ".[dev]"
```

The SQLite/NumPy index is part of the core install. To enable the local Sentence Transformers
provider, install its optional dependency group:

```bash
pip install -e ".[dev,local-embeddings]"
```

The provider does not download models by default. The default
`sentence-transformers/all-MiniLM-L6-v2` model uses a pinned commit revision. For a custom Hub model,
set `CODEBASE_RAG_EMBEDDING_MODEL` and set `CODEBASE_RAG_EMBEDDING_REVISION` to its immutable
40-character commit SHA. For a local model directory, set the revision setting to a stable version
or content fingerprint and update it whenever the model files change. Set
`CODEBASE_RAG_EMBEDDING_ALLOW_DOWNLOAD=true` only to explicitly allow model-artifact acquisition.
Repository chunks remain local.

Run checks:

```bash
ruff check .
mypy src
pytest
```

## Environment

Copy:

```bash
cp .env.example .env
```

The application must not require secrets for its default local mode.
