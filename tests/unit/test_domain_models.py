from pathlib import Path

from codebase_rag.domain.models import Citation, SourceSpan


def test_source_span_keeps_line_range() -> None:
    span = SourceSpan(start_line=10, end_line=20)
    assert span.start_line == 10
    assert span.end_line == 20


def test_citation_identifies_source() -> None:
    citation = Citation(
        relative_path=Path("src/auth.py"),
        start_line=10,
        end_line=20,
        chunk_id="chunk-1",
        symbol_name="authenticate",
    )
    assert citation.relative_path == Path("src/auth.py")
    assert citation.symbol_name == "authenticate"
