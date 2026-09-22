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
