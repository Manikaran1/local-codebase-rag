# Contributing

This repository is currently private and in foundation development.

## Development expectations

- keep modules within documented architectural boundaries
- add tests for behavior changes
- keep source citations traceable to indexed metadata
- avoid hidden network dependencies
- keep provider-specific code behind interfaces
- update architecture/ADR docs when changing major design decisions

## Local checks

```bash
ruff check .
mypy src
pytest
```

## Branch naming

Suggested:
- `feat/...`
- `fix/...`
- `docs/...`
- `refactor/...`
- `test/...`

## Commit style

Prefer small commits with intent-oriented messages, for example:

```text
feat(ingestion): add repository file filtering
test(parsing): cover python function spans
docs(adr): record vector store decision
```
