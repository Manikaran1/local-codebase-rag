"""Shared SQLite persistence for chunks, vectors, and lexical documents."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from codebase_rag.domain.models import Chunk, EmbeddingSpec, SourceSpan

SCHEMA_VERSION = 1
VECTOR_FORMAT_VERSION = 1


class IndexStorageError(RuntimeError):
    """A database could not be opened or an index schema is unsupported."""


class IndexCorruptionError(IndexStorageError):
    """Persisted index data failed structural validation."""


class IndexConfigurationError(ValueError):
    """A project index is incompatible with the requested provider settings."""


class SQLiteIndexStore:
    """Own the database schema and all SQLite transactions for indexing."""

    def __init__(self, db_path: Path | str) -> None:
        if str(db_path) == ":memory:":
            raise ValueError("Use a file-backed SQLite path so connections share persistence.")
        self.db_path = Path(db_path)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise OSError(
                f"Unable to create index directory {self.db_path.parent}: {error}"
            ) from error
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self.db_path, timeout=10.0)
        except sqlite3.Error as error:
            raise IndexStorageError(
                f"Unable to open SQLite index {self.db_path}: {error}"
            ) from error
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_metadata "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            version_row = connection.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if version_row is None:
                connection.execute(
                    "INSERT INTO schema_metadata(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif version_row["value"] != str(SCHEMA_VERSION):
                raise IndexStorageError(
                    f"SQLite index schema version {version_row['value']} is unsupported; "
                    f"this version requires {SCHEMA_VERSION}. Back up and migrate or re-index."
                )

            statements = (
                "CREATE TABLE IF NOT EXISTS chunks ("
                "chunk_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, file_id TEXT NOT NULL, "
                "relative_path TEXT NOT NULL, text TEXT NOT NULL, "
                "start_line INTEGER NOT NULL CHECK(start_line > 0), "
                "end_line INTEGER NOT NULL CHECK(end_line >= start_line), language TEXT, "
                "symbol_id TEXT, symbol_name TEXT, symbol_kind TEXT, content_hash TEXT NOT NULL, "
                "metadata_json TEXT NOT NULL)",
                "CREATE INDEX IF NOT EXISTS chunks_project_idx ON chunks(project_id)",
                "CREATE INDEX IF NOT EXISTS chunks_project_file_idx ON chunks(project_id, file_id)",
                "CREATE TABLE IF NOT EXISTS vectors ("
                "chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE, "
                "dimension INTEGER NOT NULL CHECK(dimension > 0), vector BLOB NOT NULL)",
                "CREATE TABLE IF NOT EXISTS lexical_documents ("
                "chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE, "
                "tokens_json TEXT NOT NULL, token_count INTEGER NOT NULL CHECK(token_count >= 0))",
                "CREATE TABLE IF NOT EXISTS project_index_metadata ("
                "project_id TEXT PRIMARY KEY, embedding_provider TEXT, embedding_model TEXT, "
                "vector_dimension INTEGER, vectors_normalized INTEGER "
                "CHECK(vectors_normalized IN (0, 1)), vector_format_version INTEGER, "
                "tokenizer_version TEXT, bm25_k1 REAL, bm25_b REAL)",
            )
            for statement in statements:
                connection.execute(statement)
            connection.commit()
        except sqlite3.Error as error:
            if connection.in_transaction:
                connection.rollback()
            raise IndexStorageError(
                f"Unable to initialize SQLite index {self.db_path}: {error}"
            ) from error
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except sqlite3.Error as error:
            if connection.in_transaction:
                connection.rollback()
            raise IndexStorageError(
                f"SQLite index operation failed for {self.db_path}: {error}"
            ) from error
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def ensure_embedding_metadata(
        connection: sqlite3.Connection,
        project_id: str,
        spec: EmbeddingSpec,
        dimension: int,
    ) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be greater than zero")
        SQLiteIndexStore._ensure_project_metadata(connection, project_id)
        expected: dict[str, str | int] = {
            "embedding_provider": spec.provider,
            "embedding_model": spec.model,
            "vector_dimension": dimension,
            "vectors_normalized": int(spec.normalized),
            "vector_format_version": VECTOR_FORMAT_VERSION,
        }
        SQLiteIndexStore._ensure_metadata_values(connection, project_id, expected)

    @staticmethod
    def ensure_lexical_metadata(
        connection: sqlite3.Connection,
        project_id: str,
        tokenizer_version: str,
        k1: float,
        b: float,
    ) -> None:
        SQLiteIndexStore._ensure_project_metadata(connection, project_id)
        expected: dict[str, str | float] = {
            "tokenizer_version": tokenizer_version,
            "bm25_k1": k1,
            "bm25_b": b,
        }
        SQLiteIndexStore._ensure_metadata_values(connection, project_id, expected)

    @staticmethod
    def _ensure_project_metadata(connection: sqlite3.Connection, project_id: str) -> None:
        if not project_id.strip():
            raise ValueError("project_id must not be empty")
        connection.execute(
            "INSERT OR IGNORE INTO project_index_metadata(project_id) VALUES (?)",
            (project_id,),
        )

    @staticmethod
    def _ensure_metadata_values(
        connection: sqlite3.Connection,
        project_id: str,
        expected: Mapping[str, str | int | float],
    ) -> None:
        row = connection.execute(
            "SELECT * FROM project_index_metadata WHERE project_id = ?", (project_id,)
        ).fetchone()
        if row is None:
            raise IndexCorruptionError(f"Project index metadata is missing for {project_id!r}.")
        for column, value in expected.items():
            existing = row[column]
            if existing is not None and existing != value:
                raise IndexConfigurationError(
                    f"Project {project_id!r} uses {column}={existing!r}; "
                    f"requested {column}={value!r}. Delete and rebuild the project index "
                    "to change it."
                )
            if existing is None:
                connection.execute(
                    f"UPDATE project_index_metadata SET {column} = ? WHERE project_id = ?",
                    (value, project_id),
                )

    @staticmethod
    def upsert_chunk(
        connection: sqlite3.Connection,
        chunk: Chunk,
        *,
        keep_index: Literal["vector", "lexical", "both"],
    ) -> None:
        SQLiteIndexStore._validate_chunk(chunk)
        existing = connection.execute(
            "SELECT project_id, file_id, text FROM chunks WHERE chunk_id = ?", (chunk.id,)
        ).fetchone()
        if existing is not None:
            if existing["project_id"] != chunk.project_id or existing["file_id"] != chunk.file_id:
                raise ValueError(
                    f"Chunk ID {chunk.id!r} already belongs to a different project or file."
                )
            if existing["text"] != chunk.text:
                if keep_index != "vector":
                    connection.execute("DELETE FROM vectors WHERE chunk_id = ?", (chunk.id,))
                if keep_index != "lexical":
                    connection.execute(
                        "DELETE FROM lexical_documents WHERE chunk_id = ?", (chunk.id,)
                    )

        try:
            metadata_json = json.dumps(
                chunk.metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Chunk {chunk.id!r} metadata must contain JSON-safe values."
            ) from error
        connection.execute(
            "INSERT INTO chunks (chunk_id, project_id, file_id, relative_path, text, start_line, "
            "end_line, language, symbol_id, symbol_name, symbol_kind, content_hash, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET project_id=excluded.project_id, "
            "file_id=excluded.file_id, relative_path=excluded.relative_path, text=excluded.text, "
            "start_line=excluded.start_line, end_line=excluded.end_line, "
            "language=excluded.language, "
            "symbol_id=excluded.symbol_id, symbol_name=excluded.symbol_name, "
            "symbol_kind=excluded.symbol_kind, content_hash=excluded.content_hash, "
            "metadata_json=excluded.metadata_json",
            (
                chunk.id,
                chunk.project_id,
                chunk.file_id,
                chunk.relative_path.as_posix(),
                chunk.text,
                chunk.span.start_line,
                chunk.span.end_line,
                chunk.language,
                chunk.symbol_id,
                chunk.symbol_name,
                chunk.symbol_kind,
                chunk.content_hash,
                metadata_json,
            ),
        )

    @staticmethod
    def validate_chunk(chunk: Chunk) -> None:
        SQLiteIndexStore._validate_chunk(chunk)

    @staticmethod
    def _validate_chunk(chunk: Chunk) -> None:
        if not chunk.id.strip() or not chunk.project_id.strip() or not chunk.file_id.strip():
            raise ValueError("chunk id, project_id, and file_id must not be empty")
        if chunk.span.start_line <= 0 or chunk.span.end_line < chunk.span.start_line:
            raise ValueError(f"Chunk {chunk.id!r} has an invalid source span.")

    @staticmethod
    def write_vector(
        connection: sqlite3.Connection,
        chunk_id: str,
        dimension: int,
        vector_blob: bytes,
    ) -> None:
        connection.execute(
            "INSERT INTO vectors(chunk_id, dimension, vector) VALUES (?, ?, ?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET dimension=excluded.dimension, "
            "vector=excluded.vector",
            (chunk_id, dimension, vector_blob),
        )

    @staticmethod
    def write_tokens(
        connection: sqlite3.Connection,
        chunk_id: str,
        tokens: Sequence[str],
    ) -> None:
        tokens_json = json.dumps(list(tokens), ensure_ascii=False, separators=(",", ":"))
        connection.execute(
            "INSERT INTO lexical_documents(chunk_id, tokens_json, token_count) VALUES (?, ?, ?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET tokens_json=excluded.tokens_json, "
            "token_count=excluded.token_count",
            (chunk_id, tokens_json, len(tokens)),
        )

    @staticmethod
    def chunk_from_row(row: sqlite3.Row) -> Chunk:
        try:
            metadata = json.loads(row["metadata_json"])
        except (json.JSONDecodeError, TypeError) as error:
            raise IndexCorruptionError(
                f"Chunk {row['chunk_id']!r} contains malformed metadata JSON."
            ) from error
        if not isinstance(metadata, dict):
            raise IndexCorruptionError(f"Chunk {row['chunk_id']!r} metadata is not a JSON object.")
        return Chunk(
            id=row["chunk_id"],
            project_id=row["project_id"],
            file_id=row["file_id"],
            relative_path=Path(row["relative_path"]),
            text=row["text"],
            span=SourceSpan(row["start_line"], row["end_line"]),
            language=row["language"],
            symbol_id=row["symbol_id"],
            symbol_name=row["symbol_name"],
            symbol_kind=row["symbol_kind"],
            content_hash=row["content_hash"],
            metadata=metadata,
        )

    def vector_rows(self, project_id: str) -> list[sqlite3.Row]:
        with self.read_connection() as connection:
            return list(
                connection.execute(
                    "SELECT c.*, v.dimension, v.vector FROM vectors AS v "
                    "JOIN chunks AS c ON c.chunk_id = v.chunk_id "
                    "WHERE c.project_id = ? ORDER BY c.chunk_id ASC",
                    (project_id,),
                ).fetchall()
            )

    def lexical_rows(self, project_id: str) -> list[sqlite3.Row]:
        with self.read_connection() as connection:
            return list(
                connection.execute(
                    "SELECT c.*, l.tokens_json, l.token_count FROM lexical_documents AS l "
                    "JOIN chunks AS c ON c.chunk_id = l.chunk_id "
                    "WHERE c.project_id = ? ORDER BY c.chunk_id ASC",
                    (project_id,),
                ).fetchall()
            )

    def project_metadata(self, project_id: str) -> dict[str, object] | None:
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM project_index_metadata WHERE project_id = ?", (project_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def delete_file(self, project_id: str, file_id: str) -> int:
        if not project_id.strip() or not file_id.strip():
            raise ValueError("project_id and file_id must not be empty")
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM chunks WHERE project_id = ? AND file_id = ?",
                (project_id, file_id),
            )
            return cursor.rowcount

    def delete_project(self, project_id: str) -> int:
        if not project_id.strip():
            raise ValueError("project_id must not be empty")
        with self.transaction() as connection:
            cursor = connection.execute("DELETE FROM chunks WHERE project_id = ?", (project_id,))
            deleted = cursor.rowcount
            connection.execute(
                "DELETE FROM project_index_metadata WHERE project_id = ?", (project_id,)
            )
            return deleted

    def replace_file(
        self,
        *,
        project_id: str,
        file_id: str,
        chunks: list[Chunk],
        vectors: list[bytes],
        dimension: int | None,
        tokens: list[list[str]],
        embedding_spec: EmbeddingSpec,
        tokenizer_version: str,
        bm25_k1: float,
        bm25_b: float,
    ) -> None:
        if not project_id.strip() or not file_id.strip():
            raise ValueError("project_id and file_id must not be empty")
        if len(chunks) != len(vectors) or len(chunks) != len(tokens):
            raise ValueError("chunks, vectors, and token lists must have equal lengths")
        if chunks and (dimension is None or dimension <= 0):
            raise ValueError("a positive vector dimension is required when indexing chunks")
        for chunk, vector_blob in zip(chunks, vectors, strict=True):
            self._validate_chunk(chunk)
            if chunk.project_id != project_id or chunk.file_id != file_id:
                raise ValueError(
                    "all replacement chunks must belong to the requested project and file"
                )
            if dimension is not None and len(vector_blob) != dimension * 4:
                raise ValueError(
                    f"Chunk {chunk.id!r} has a vector blob with an invalid byte length."
                )

        with self.transaction() as connection:
            if chunks and dimension is not None:
                self.ensure_embedding_metadata(connection, project_id, embedding_spec, dimension)
                self.ensure_lexical_metadata(
                    connection, project_id, tokenizer_version, bm25_k1, bm25_b
                )
            connection.execute(
                "DELETE FROM chunks WHERE project_id = ? AND file_id = ?", (project_id, file_id)
            )
            if dimension is None:
                return
            for chunk, vector_blob, document_tokens in zip(chunks, vectors, tokens, strict=True):
                self.upsert_chunk(connection, chunk, keep_index="both")
                self.write_vector(connection, chunk.id, dimension, vector_blob)
                self.write_tokens(connection, chunk.id, document_tokens)
