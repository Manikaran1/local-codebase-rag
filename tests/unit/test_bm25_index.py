from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from codebase_rag.domain.models import Chunk, SourceSpan
from codebase_rag.index.bm25 import SQLiteBM25Index, tokenize
from codebase_rag.index.storage import (
    IndexConfigurationError,
    IndexCorruptionError,
    SQLiteIndexStore,
)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    project_id: str = "project-a",
    file_id: str = "file-a",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        project_id=project_id,
        file_id=file_id,
        relative_path=Path(f"src/{file_id}.py"),
        text=text,
        span=SourceSpan(1, 2),
        language="Python",
        content_hash=f"hash-{chunk_id}",
        metadata={"kind": "function", "labels": ["auth", "lookup"]},
    )


def _index(tmp_path: Path, *, k1: float = 1.5, b: float = 0.75) -> SQLiteBM25Index:
    return SQLiteBM25Index(SQLiteIndexStore(tmp_path / "index.sqlite3"), k1=k1, b=b)


def test_tokenizer_splits_snake_camel_pascal_numbers_and_casefolds() -> None:
    assert tokenize("HTTPServer findUserByID user_name 404Error") == [
        "httpserver",
        "http",
        "server",
        "finduserbyid",
        "find",
        "user",
        "by",
        "id",
        "user_name",
        "user",
        "name",
        "404error",
        "404",
        "error",
    ]


def test_search_returns_raw_bm25_scores_and_project_scoped_hits(tmp_path: Path) -> None:
    index = _index(tmp_path)
    chunks = [
        _chunk("long", "lookup request validation with extra context"),
        _chunk("short", "lookup request"),
        _chunk("unrelated", "database connection pool"),
        _chunk("other-project", "lookup request", project_id="project-b"),
    ]
    index.upsert(chunks)

    hits = index.search("project-a", "LOOKUP request", limit=10)

    assert [hit.chunk.id for hit in hits] == ["short", "long"]
    assert hits[0].lexical_score is not None and hits[0].lexical_score > 0
    for rank, hit in enumerate(hits, start=1):
        assert hit.rank == rank
        assert hit.vector_score is None
        assert hit.fusion_score is None
        assert hit.rerank_score is None


def test_identifier_terms_retrieve_exact_code_and_prose(tmp_path: Path) -> None:
    index = _index(tmp_path)
    code = _chunk("code", "def load_user_profile(): return userId")
    prose = _chunk(
        "prose", "The loader validates a profile before returning it.", file_id="guide.md"
    )
    index.upsert([code, prose])

    snake_hits = index.search("project-a", "load_user_profile", limit=5)
    camel_hits = index.search("project-a", "userId", limit=5)
    prose_hits = index.search("project-a", "validates profile", limit=5)

    assert snake_hits[0].chunk.id == "code"
    assert [hit.chunk.id for hit in camel_hits] == ["code"]
    assert prose_hits[0].chunk.id == "prose"


def test_ties_use_chunk_id_and_reopen_persistence_keeps_complete_chunks(tmp_path: Path) -> None:
    index = _index(tmp_path)
    chunks = [
        _chunk("chunk-c", "exact token"),
        _chunk("chunk-a", "exact token"),
        _chunk("chunk-b", "exact token"),
    ]
    index.upsert(chunks)

    reopened = _index(tmp_path)
    first = reopened.search("project-a", "exact token", limit=2)
    second = reopened.search("project-a", "exact token", limit=3)

    assert [hit.chunk.id for hit in first] == ["chunk-a", "chunk-b"]
    assert [hit.chunk.id for hit in second] == ["chunk-a", "chunk-b", "chunk-c"]
    assert second[0].chunk == next(chunk for chunk in chunks if chunk.id == "chunk-a")
    assert first[0].lexical_score == pytest.approx(second[0].lexical_score)


def test_same_chunk_id_upsert_replaces_lexical_content(tmp_path: Path) -> None:
    index = _index(tmp_path)
    original = _chunk("same", "oldterm")
    updated = replace(original, text="newterm", metadata={"updated": True})
    index.upsert([original])
    index.upsert([updated])

    assert index.search("project-a", "oldterm", 5) == []
    hits = index.search("project-a", "newterm", 5)
    assert len(hits) == 1
    assert hits[0].chunk == updated


def test_file_and_project_deletion_remove_lexical_rows(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.upsert(
        [
            _chunk("a1", "token alpha", file_id="a.py"),
            _chunk("a2", "token beta", file_id="a.py"),
            _chunk("b1", "token gamma", file_id="b.py"),
            _chunk("other", "token delta", project_id="project-b"),
        ]
    )

    assert index.delete_file("project-a", "a.py") == 2
    assert [hit.chunk.id for hit in index.search("project-a", "token", 10)] == ["b1"]
    assert index.delete_project("project-a") == 1
    assert index.search("project-a", "token", 10) == []
    assert [hit.chunk.id for hit in index.search("project-b", "token", 10)] == ["other"]


def test_empty_query_limit_and_configuration_validation(tmp_path: Path) -> None:
    index = _index(tmp_path)

    assert index.search("empty-project", "---", limit=2) == []
    assert index.search("empty-project", "words", limit=2) == []
    with pytest.raises(ValueError, match="limit must be an integer greater than zero"):
        index.search("empty-project", "words", limit=0)
    with pytest.raises(ValueError, match="BM25 k1"):
        SQLiteBM25Index(index.store, k1=0)
    with pytest.raises(ValueError, match="BM25 b"):
        SQLiteBM25Index(index.store, b=1.5)

    index.upsert([_chunk("configured", "configured term")])
    other_config = SQLiteBM25Index(index.store, k1=2.0)
    with pytest.raises(IndexConfigurationError, match="bm25_k1"):
        other_config.search("project-a", "configured", 2)


def test_corrupt_token_json_is_reported(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.upsert([_chunk("corrupt", "token")])
    with index.store.transaction() as connection:
        connection.execute(
            "UPDATE lexical_documents SET tokens_json = ? WHERE chunk_id = ?",
            ("{not-json", "corrupt"),
        )

    with pytest.raises(IndexCorruptionError, match="malformed token JSON"):
        index.search("project-a", "token", 1)


def test_missing_project_metadata_is_reported_for_existing_lexical_rows(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    index.upsert([_chunk("corrupt", "token")])
    with index.store.transaction() as connection:
        connection.execute(
            "DELETE FROM project_index_metadata WHERE project_id = ?", ("project-a",)
        )

    with pytest.raises(IndexCorruptionError, match="metadata is missing"):
        index.search("project-a", "token", 1)
