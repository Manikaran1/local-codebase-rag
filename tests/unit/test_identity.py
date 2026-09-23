from codebase_rag.domain.identity import chunk_id_for, content_hash_for, symbol_id_for
from codebase_rag.domain.models import SourceSpan


def test_chunk_identity_tracks_file_span_and_exact_content() -> None:
    span = SourceSpan(4, 5)
    base = chunk_id_for(
        file_id="file-one",
        span=span,
        text="return 1\n",
        symbol_id="symbol-one",
        metadata={"chunk_kind": "structural"},
    )

    assert base == chunk_id_for(
        file_id="file-one",
        span=span,
        text="return 1\n",
        symbol_id="symbol-one",
        metadata={"chunk_kind": "structural"},
    )
    assert base != chunk_id_for(
        file_id="file-two",
        span=span,
        text="return 1\n",
        symbol_id="symbol-one",
        metadata={"chunk_kind": "structural"},
    )
    assert base != chunk_id_for(
        file_id="file-one",
        span=SourceSpan(5, 6),
        text="return 1\n",
        symbol_id="symbol-one",
        metadata={"chunk_kind": "structural"},
    )
    assert base != chunk_id_for(
        file_id="file-one",
        span=span,
        text="return 2\n",
        symbol_id="symbol-one",
        metadata={"chunk_kind": "structural"},
    )
    assert content_hash_for("return 1\n") != content_hash_for("return 2\n")
    assert symbol_id_for("file-one", "Example.run", "method", span) != symbol_id_for(
        "file-two",
        "Example.run",
        "method",
        span,
    )
