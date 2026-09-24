from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from codebase_rag.domain.models import Chunk, SourceSpan
from codebase_rag.index.service import IndexingService
from codebase_rag.index.storage import SQLiteIndexStore
from tests.fakes import DeterministicFakeEmbedder


def _chunk(chunk_id: str, text: str, *, file_id: str = "file-a") -> Chunk:
    return Chunk(
        id=chunk_id,
        project_id="project-a",
        file_id=file_id,
        relative_path=Path(f"src/{file_id}.py"),
        text=text,
        span=SourceSpan(1, 1),
        language="Python",
        content_hash=f"hash-{chunk_id}",
        metadata={"function": "run"},
    )


def _service(tmp_path: Path) -> IndexingService:
    return IndexingService(
        store=SQLiteIndexStore(tmp_path / "index.sqlite3"),
        embedder=DeterministicFakeEmbedder(dimension=8),
    )


def test_file_replacement_updates_vector_and_lexical_indexes_together(tmp_path: Path) -> None:
    service = _service(tmp_path)
    old_chunks = [_chunk("old-one", "legacy_endpoint"), _chunk("old-two", "remove this handler")]
    service.index_file("project-a", "file-a", old_chunks)

    new_chunks = [_chunk("new-one", "replacement_endpoint")]
    service.index_file("project-a", "file-a", new_chunks)

    vector_ids = {row["chunk_id"] for row in service.store.vector_rows("project-a")}
    lexical_ids = {row["chunk_id"] for row in service.store.lexical_rows("project-a")}
    assert vector_ids == {"new-one"}
    assert lexical_ids == {"new-one"}
    assert (
        service.vector_index.search(
            "project-a", service.embedder.embed_query("replacement_endpoint"), 5
        )[0].chunk.id
        == "new-one"
    )
    assert [
        hit.chunk.id for hit in service.lexical_index.search("project-a", "replacement_endpoint", 5)
    ] == ["new-one"]


def test_file_replacement_is_atomic_when_chunk_metadata_cannot_be_serialized(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    original = _chunk("original", "keep this record")
    service.index_file("project-a", "file-a", [original])
    invalid = replace(
        _chunk("invalid", "replacement text"),
        metadata={"not_json_safe": object()},
    )

    with pytest.raises(ValueError, match="JSON-safe"):
        service.index_file("project-a", "file-a", [invalid])

    assert {row["chunk_id"] for row in service.store.vector_rows("project-a")} == {"original"}
    assert {row["chunk_id"] for row in service.store.lexical_rows("project-a")} == {"original"}
    assert (
        service.vector_index.search(
            "project-a", service.embedder.embed_query("keep this record"), 1
        )[0].chunk
        == original
    )


def test_empty_file_replacement_clears_both_indexes(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.index_file("project-a", "file-a", [_chunk("old", "old text")])

    service.index_file("project-a", "file-a", [])

    assert service.store.vector_rows("project-a") == []
    assert service.store.lexical_rows("project-a") == []
