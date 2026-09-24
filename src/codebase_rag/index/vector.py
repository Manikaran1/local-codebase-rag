"""SQLite-backed exact cosine vector index."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from codebase_rag.domain.models import Chunk, EmbeddingSpec, SearchHit
from codebase_rag.index.base import VectorIndex
from codebase_rag.index.storage import (
    VECTOR_FORMAT_VERSION,
    IndexConfigurationError,
    IndexCorruptionError,
    SQLiteIndexStore,
)


class SQLiteVectorIndex(VectorIndex):
    """Persist float32 embeddings and search them with exact cosine similarity."""

    def __init__(self, store: SQLiteIndexStore, embedding_spec: EmbeddingSpec) -> None:
        if (
            not embedding_spec.provider.strip()
            or not embedding_spec.model.strip()
            or not embedding_spec.revision.strip()
        ):
            raise ValueError("embedding provider, model, and revision must not be empty")
        self.store = store
        self.embedding_spec = embedding_spec

    @classmethod
    def from_data_dir(cls, data_dir: Path, embedding_spec: EmbeddingSpec) -> SQLiteVectorIndex:
        return cls(SQLiteIndexStore(data_dir / "indexes" / "index.sqlite3"), embedding_spec)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        dimension, vector_blobs = self.prepare_vectors(chunks, vectors)
        if not chunks:
            return
        with self.store.transaction() as connection:
            for chunk, vector_blob in zip(chunks, vector_blobs, strict=True):
                self.store.ensure_embedding_metadata(
                    connection, chunk.project_id, self.embedding_spec, dimension
                )
                self.store.upsert_chunk(connection, chunk, keep_index="vector")
                self.store.write_vector(connection, chunk.id, dimension, vector_blob)

    def search(self, project_id: str, vector: list[float], limit: int) -> list[SearchHit]:
        _validate_project_and_limit(project_id, limit)
        query = _as_vector(vector, "query vector")
        query_norm = float(np.linalg.norm(query.astype(np.float64)))
        rows = self.store.vector_rows(project_id)
        metadata = self.store.project_metadata(project_id)
        if metadata is None:
            if rows:
                raise IndexCorruptionError(
                    f"Project vector metadata is missing for {project_id!r}."
                )
            return []
        if metadata.get("vector_dimension") is None:
            if rows:
                raise IndexCorruptionError(
                    f"Project vector metadata is incomplete for {project_id!r}."
                )
            return []
        _check_embedding_metadata(metadata, self.embedding_spec, int(query.size))
        if not rows:
            return []
        vectors: list[np.ndarray] = []
        chunks: list[Chunk] = []
        for row in rows:
            dimension = row["dimension"]
            blob = bytes(row["vector"])
            if dimension != query.size or len(blob) != dimension * 4:
                raise IndexCorruptionError(
                    f"Stored vector for chunk {row['chunk_id']!r} has incompatible dimensions "
                    "or byte length."
                )
            stored = np.frombuffer(blob, dtype="<f4")
            if stored.size != dimension or not np.isfinite(stored).all():
                raise IndexCorruptionError(
                    f"Stored vector for chunk {row['chunk_id']!r} is malformed or non-finite."
                )
            stored_norm = float(np.linalg.norm(stored.astype(np.float64)))
            if stored_norm == 0:
                raise IndexCorruptionError(f"Stored vector for chunk {row['chunk_id']!r} is zero.")
            if self.embedding_spec.normalized and not math.isclose(
                stored_norm, 1.0, rel_tol=1e-3, abs_tol=1e-3
            ):
                raise IndexCorruptionError(
                    f"Stored vector for chunk {row['chunk_id']!r} is not normalized as configured."
                )
            vectors.append(stored)
            chunks.append(self.store.chunk_from_row(row))

        matrix = np.vstack(vectors).astype(np.float64)
        query64 = query.astype(np.float64)
        scores = (matrix @ query64) / (np.linalg.norm(matrix, axis=1) * query_norm)
        bounded_scores = np.clip(scores, -1.0, 1.0)
        ranked = sorted(
            zip(chunks, bounded_scores.tolist(), strict=True),
            key=lambda item: (-item[1], item[0].id),
        )[:limit]
        return [
            SearchHit(chunk=chunk, rank=rank, vector_score=score)
            for rank, (chunk, score) in enumerate(ranked, start=1)
        ]

    def delete_file(self, project_id: str, file_id: str) -> int:
        return self.store.delete_file(project_id, file_id)

    def delete_project(self, project_id: str) -> int:
        return self.store.delete_project(project_id)

    def prepare_vectors(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> tuple[int, list[bytes]]:
        """Validate and encode vectors before an atomic file replacement."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal lengths")
        if not chunks:
            return 0, []
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique in one vector upsert")

        prepared: list[bytes] = []
        dimension: int | None = None
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.store.validate_chunk(chunk)
            values = _as_vector(vector, f"vector for chunk {chunk.id!r}")
            if dimension is None:
                dimension = int(values.size)
            elif values.size != dimension:
                raise ValueError("all vectors in one upsert must have the same dimension")
            norm = float(np.linalg.norm(values.astype(np.float64)))
            if norm == 0:
                raise ValueError(f"vector for chunk {chunk.id!r} must not be zero")
            if self.embedding_spec.normalized and not math.isclose(
                norm, 1.0, rel_tol=1e-3, abs_tol=1e-3
            ):
                raise ValueError(
                    f"vector for chunk {chunk.id!r} must be normalized for this embedding provider"
                )
            prepared.append(values.astype("<f4").tobytes(order="C"))
        if dimension is None:
            raise ValueError("at least one vector is required")
        return dimension, prepared


def _as_vector(values: Sequence[float], label: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must contain finite numeric values") from error
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain finite numeric values")
    if float(np.linalg.norm(vector.astype(np.float64))) == 0:
        raise ValueError(f"{label} must not be zero")
    return vector


def _validate_project_and_limit(project_id: str, limit: int) -> None:
    if not project_id.strip():
        raise ValueError("project_id must not be empty")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be an integer greater than zero")


def _check_embedding_metadata(
    metadata: dict[str, object],
    spec: EmbeddingSpec,
    dimension: int,
) -> None:
    stored_identity = {
        "provider": metadata.get("embedding_provider"),
        "model": metadata.get("embedding_model"),
        "revision": metadata.get("embedding_revision"),
    }
    requested_identity = {
        "provider": spec.provider,
        "model": spec.model,
        "revision": spec.revision,
    }
    for key, value in stored_identity.items():
        if value is None:
            raise IndexCorruptionError(f"Project vector metadata is missing embedding {key!r}.")
    if stored_identity != requested_identity:
        raise IndexConfigurationError(
            f"Project index uses embedding identity provider={stored_identity['provider']!r}, "
            f"model={stored_identity['model']!r}, revision={stored_identity['revision']!r}; "
            f"requested provider={spec.provider!r}, model={spec.model!r}, "
            f"revision={spec.revision!r}. Rebuild the project index to change embedding space."
        )

    expected: dict[str, object] = {
        "vector_dimension": dimension,
        "vectors_normalized": int(spec.normalized),
        "vector_format_version": VECTOR_FORMAT_VERSION,
    }
    for key, value in expected.items():
        stored = metadata.get(key)
        if stored is None:
            raise IndexCorruptionError(f"Project vector metadata is missing {key!r}.")
        if stored != value:
            raise IndexConfigurationError(
                f"Project index uses {key}={stored!r}; requested {key}={value!r}."
            )
