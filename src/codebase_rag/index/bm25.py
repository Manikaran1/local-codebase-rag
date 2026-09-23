"""SQLite-backed BM25 lexical index and deterministic code-aware tokenizer."""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from codebase_rag.domain.models import Chunk, SearchHit
from codebase_rag.index.base import LexicalIndex
from codebase_rag.index.storage import (
    IndexConfigurationError,
    IndexCorruptionError,
    SQLiteIndexStore,
)

TOKENIZER_VERSION = "casefold-identifier-v1"
_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_ACRONYM_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")


def tokenize(text: str) -> list[str]:
    """Preserve identifiers and add case-folded parts for code-aware matching."""
    tokens: list[str] = []
    for raw_token in _TOKEN_PATTERN.findall(text):
        if not raw_token.strip("_"):
            continue
        tokens.append(raw_token.casefold())
        if "_" in raw_token:
            tokens.extend(part.casefold() for part in raw_token.split("_") if part)
            continue
        split = _ACRONYM_BOUNDARY.sub(r"\1 \2", raw_token)
        split = _CAMEL_BOUNDARY.sub(r"\1 \2", split)
        parts = split.split()
        if len(parts) > 1:
            tokens.extend(part.casefold() for part in parts)
    return tokens


class SQLiteBM25Index(LexicalIndex):
    """Persist tokenized chunk text and rank it with BM25 (k1=1.5, b=0.75)."""

    def __init__(
        self,
        store: SQLiteIndexStore,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0 or not math.isfinite(k1):
            raise ValueError("BM25 k1 must be a finite value greater than zero")
        if not 0 <= b <= 1 or not math.isfinite(b):
            raise ValueError("BM25 b must be between zero and one")
        self.store = store
        self.k1 = float(k1)
        self.b = float(b)

    def upsert(self, chunks: list[Chunk]) -> None:
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique in one lexical upsert")
        prepared = [(chunk, tokenize(chunk.text)) for chunk in chunks]
        for chunk, _ in prepared:
            self.store.validate_chunk(chunk)
        if not prepared:
            return

        with self.store.transaction() as connection:
            for chunk, tokens in prepared:
                self.store.ensure_lexical_metadata(
                    connection, chunk.project_id, TOKENIZER_VERSION, self.k1, self.b
                )
                self.store.upsert_chunk(connection, chunk, keep_index="lexical")
                self.store.write_tokens(connection, chunk.id, tokens)

    def search(self, project_id: str, query: str, limit: int) -> list[SearchHit]:
        if not project_id.strip():
            raise ValueError("project_id must not be empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be an integer greater than zero")
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        rows = self.store.lexical_rows(project_id)
        metadata = self.store.project_metadata(project_id)
        if metadata is None:
            if rows:
                raise IndexCorruptionError(
                    f"Project lexical metadata is missing for {project_id!r}."
                )
            return []
        if metadata.get("tokenizer_version") is None:
            if rows:
                raise IndexCorruptionError(
                    f"Project lexical metadata is incomplete for {project_id!r}."
                )
            return []
        self._check_metadata(metadata)
        if not rows:
            return []

        documents: list[tuple[Chunk, list[str]]] = []
        document_frequency: Counter[str] = Counter()
        total_length = 0
        for row in rows:
            try:
                tokens = json.loads(row["tokens_json"])
            except (json.JSONDecodeError, TypeError) as error:
                raise IndexCorruptionError(
                    f"Lexical record for chunk {row['chunk_id']!r} has malformed token JSON."
                ) from error
            if (
                not isinstance(tokens, list)
                or any(not isinstance(token, str) for token in tokens)
                or row["token_count"] != len(tokens)
            ):
                raise IndexCorruptionError(
                    f"Lexical record for chunk {row['chunk_id']!r} has invalid token data."
                )
            typed_tokens = list(tokens)
            documents.append((self.store.chunk_from_row(row), typed_tokens))
            total_length += len(typed_tokens)
            document_frequency.update(set(typed_tokens))

        document_count = len(documents)
        average_length = total_length / document_count
        if average_length == 0:
            return []
        query_frequency = Counter(query_tokens)
        scored: list[tuple[Chunk, float]] = []
        for chunk, tokens in documents:
            frequencies = Counter(tokens)
            document_length = len(tokens)
            score = 0.0
            for term, query_count in query_frequency.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                term_document_frequency = document_frequency[term]
                inverse_frequency = math.log(
                    1.0
                    + (document_count - term_document_frequency + 0.5)
                    / (term_document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * document_length / average_length
                )
                score += (
                    query_count * inverse_frequency * (frequency * (self.k1 + 1.0) / denominator)
                )
            if score > 0:
                scored.append((chunk, score))

        ranked = sorted(scored, key=lambda item: (-item[1], item[0].id))[:limit]
        return [
            SearchHit(chunk=chunk, rank=rank, lexical_score=score)
            for rank, (chunk, score) in enumerate(ranked, start=1)
        ]

    def delete_file(self, project_id: str, file_id: str) -> int:
        return self.store.delete_file(project_id, file_id)

    def delete_project(self, project_id: str) -> int:
        return self.store.delete_project(project_id)

    def _check_metadata(self, metadata: dict[str, object]) -> None:
        expected: dict[str, object] = {
            "tokenizer_version": TOKENIZER_VERSION,
            "bm25_k1": self.k1,
            "bm25_b": self.b,
        }
        for key, value in expected.items():
            stored = metadata.get(key)
            if stored is None:
                raise IndexCorruptionError(f"Project lexical metadata is missing {key!r}.")
            if stored != value:
                raise IndexConfigurationError(
                    f"Project lexical index uses {key}={stored!r}; requested {key}={value!r}. "
                    "Delete and rebuild the project index to change it."
                )
