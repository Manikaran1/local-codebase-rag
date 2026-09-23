from codebase_rag.domain.identity import chunk_id_for, content_hash_for, symbol_id_for
from codebase_rag.domain.models import SourceSpan


def test_chunk_identity_tracks_file_span_and_exact_content() -> None:
    span = SourceSpan(4, 5)
    base = chunk_id_for(file_id="file-one", span=span, text="return 1\n")

    assert base == chunk_id_for(file_id="file-one", span=span, text="return 1\n")
    assert base != chunk_id_for(file_id="file-two", span=span, text="return 1\n")
    assert base != chunk_id_for(
        file_id="file-one",
        span=SourceSpan(5, 6),
        text="return 1\n",
    )
    assert base != chunk_id_for(file_id="file-one", span=span, text="return 2\n")
    assert content_hash_for("return 1\n") != content_hash_for("return 2\n")


def test_chunk_identity_uses_source_offset_only_to_disambiguate_same_span_text() -> None:
    span = SourceSpan(1, 1)
    first = chunk_id_for(
        file_id="file-one",
        span=span,
        text="same",
        source_offset=10,
    )
    second = chunk_id_for(
        file_id="file-one",
        span=span,
        text="same",
        source_offset=15,
    )

    assert first == chunk_id_for(
        file_id="file-one",
        span=span,
        text="same",
        source_offset=10,
    )
    assert first != second


def test_symbol_identity_remains_independent_and_deterministic() -> None:
    span = SourceSpan(4, 5)

    assert symbol_id_for("file-one", "Example.run", "method", span) == symbol_id_for(
        "file-one",
        "Example.run",
        "method",
        span,
    )
    assert symbol_id_for("file-one", "Example.run", "method", span) != symbol_id_for(
        "file-two",
        "Example.run",
        "method",
        span,
    )
