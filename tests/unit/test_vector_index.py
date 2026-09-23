from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from codebase_rag.domain.models import Chunk, EmbeddingSpec, SourceSpan
from codebase_rag.index.storage import (
    IndexConfigurationError,
    IndexCorruptionError,
    SQLiteIndexStore,
)
from codebase_rag.index.vector import SQLiteVectorIndex


def _chunk(
    chunk_id: str,
    *,
    project_id: str = "project-a",
    file_id: str = "file-a",
    text: str | None = None,
) -> Chunk:
    content = text if text is not None else f"text for {chunk_id}"
    return Chunk(
        id=chunk_id,
        project_id=project_id,
        file_id=file_id,
        relative_path=Path(f"src/{file_id}.py"),
        text=content,
        span=SourceSpan(3, 4),
        language="Python",
        symbol_id=f"symbol-{chunk_id}",
        symbol_name="Store.lookup",
        symbol_kind="method",
        content_hash=f"hash-{chunk_id}",
        metadata={"z": [1, 2], "heading": "Lookup"},
    )


def _index(tmp_path: Path, spec: EmbeddingSpec | None = None) -> SQLiteVectorIndex:
    resolved_spec = spec or EmbeddingSpec("fake", "model-a", True)
    return SQLiteVectorIndex(SQLiteIndexStore(tmp_path / "index.sqlite3"), resolved_spec)


def test_search_is_project_scoped_and_returns_raw_vector_scores(tmp_path: Path) -> None:
    index = _index(tmp_path)
    chunks = [
        _chunk("chunk-z"),
        _chunk("chunk-low"),
        _chunk("chunk-a"),
        _chunk("other-project", project_id="project-b"),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
    index.upsert(chunks, vectors)

    hits = index.search("project-a", [1.0, 0.0], limit=10)

    assert [hit.chunk.id for hit in hits] == ["chunk-a", "chunk-z", "chunk-low"]
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert [hit.vector_score for hit in hits] == pytest.approx([1.0, 1.0, 0.0])
    for hit in hits:
        assert hit.fusion_score is None
        assert hit.lexical_score is None
        assert hit.rerank_score is None


def test_search_reconstructs_the_complete_chunk_after_reopening(tmp_path: Path) -> None:
    chunk = _chunk("reconstruct", text="def lookup(): return 1")
    index = _index(tmp_path)
    index.upsert([chunk], [[0.6, 0.8]])

    reopened = _index(tmp_path)
    hits = reopened.search("project-a", [0.6, 0.8], limit=1)

    assert len(hits) == 1
    assert hits[0].chunk == chunk
    assert hits[0].vector_score == pytest.approx(1.0)


def test_schema_and_project_metadata_record_versions_and_float32_storage(tmp_path: Path) -> None:
    index = _index(tmp_path)
    chunk = _chunk("metadata", text="stable metadata")
    index.upsert([chunk], [[0.6, 0.8]])

    metadata = index.store.project_metadata("project-a")
    assert metadata is not None
    assert metadata["embedding_provider"] == "fake"
    assert metadata["embedding_model"] == "model-a"
    assert metadata["vector_dimension"] == 2
    assert metadata["vectors_normalized"] == 1
    assert metadata["vector_format_version"] == 1
    with index.store.read_connection() as connection:
        schema = connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()
        record = connection.execute(
            "SELECT dimension, vector FROM vectors WHERE chunk_id = ?", (chunk.id,)
        ).fetchone()
        index_names = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }

    assert schema["value"] == "1"
    assert record["dimension"] == 2
    assert len(record["vector"]) == 8
    assert {"chunks_project_idx", "chunks_project_file_idx"} <= index_names
    assert chunk.metadata == {"z": [1, 2], "heading": "Lookup"}


def test_empty_index_limit_and_vector_validation_policy(tmp_path: Path) -> None:
    index = _index(tmp_path)

    assert index.search("empty-project", [1.0, 0.0], limit=3) == []
    with pytest.raises(ValueError, match="limit must be an integer greater than zero"):
        index.search("empty-project", [1.0, 0.0], limit=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        index.search("empty-project", [[1.0, 0.0]], limit=1)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite"):
        index.search("empty-project", [float("nan"), 0.0], limit=1)
    with pytest.raises(ValueError, match="equal lengths"):
        index.upsert([_chunk("mismatch")], [])


def test_dimension_and_embedding_model_mismatches_are_rejected_without_replacing_data(
    tmp_path: Path,
) -> None:
    chunk = _chunk("stable")
    first = _index(tmp_path)
    first.upsert([chunk], [[1.0, 0.0]])

    with pytest.raises(IndexConfigurationError, match="vector_dimension"):
        first.upsert([chunk], [[1.0, 0.0, 0.0]])
    other_model = _index(tmp_path, EmbeddingSpec("fake", "model-b", True))
    with pytest.raises(IndexConfigurationError, match="embedding_model"):
        other_model.upsert([chunk], [[0.0, 1.0]])

    assert first.search("project-a", [1.0, 0.0], limit=1)[0].chunk == chunk
    with pytest.raises(IndexConfigurationError, match="vector_dimension"):
        first.search("project-a", [1.0, 0.0, 0.0], limit=1)


def test_same_chunk_id_upsert_replaces_vector_and_chunk_data(tmp_path: Path) -> None:
    index = _index(tmp_path)
    original = _chunk("same", text="before")
    replacement = replace(original, text="after", metadata={"updated": True})
    index.upsert([original], [[1.0, 0.0]])
    index.upsert([replacement], [[0.0, 1.0]])

    hit = index.search("project-a", [0.0, 1.0], limit=1)[0]

    assert hit.chunk == replacement
    assert hit.vector_score == pytest.approx(1.0)


def test_file_and_project_deletion_remove_persisted_vectors(tmp_path: Path) -> None:
    index = _index(tmp_path)
    chunks = [
        _chunk("a1", file_id="a.py"),
        _chunk("a2", file_id="a.py"),
        _chunk("b1", file_id="b.py"),
        _chunk("other", project_id="project-b", file_id="a.py"),
    ]
    index.upsert(chunks, [[1.0, 0.0] for _ in chunks])

    assert index.delete_file("project-a", "a.py") == 2
    assert [hit.chunk.id for hit in index.search("project-a", [1.0, 0.0], 10)] == ["b1"]
    assert index.delete_project("project-a") == 1
    assert index.search("project-a", [1.0, 0.0], 10) == []
    assert [hit.chunk.id for hit in index.search("project-b", [1.0, 0.0], 10)] == ["other"]


def test_corrupted_vector_blob_is_reported(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.upsert([_chunk("corrupt")], [[1.0, 0.0]])
    with index.store.transaction() as connection:
        connection.execute(
            "UPDATE vectors SET vector = ? WHERE chunk_id = ?", (b"short", "corrupt")
        )

    with pytest.raises(IndexCorruptionError, match="byte length"):
        index.search("project-a", [1.0, 0.0], 1)


def test_missing_project_metadata_is_reported_for_existing_vectors(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    index.upsert([_chunk("corrupt")], [[1.0, 0.0]])
    with index.store.transaction() as connection:
        connection.execute(
            "DELETE FROM project_index_metadata WHERE project_id = ?", ("project-a",)
        )

    with pytest.raises(IndexCorruptionError, match="metadata is missing"):
        index.search("project-a", [1.0, 0.0], 1)
