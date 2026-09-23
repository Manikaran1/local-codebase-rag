from pathlib import Path

from codebase_rag.chunking.code import CodeChunker
from codebase_rag.chunking.prose import ProseChunker
from codebase_rag.config.settings import Settings
from codebase_rag.domain.identity import content_hash_for
from codebase_rag.domain.models import SourceFile, SourceSpan
from codebase_rag.parsing.dispatcher import ParserDispatcher


def _source(path: str, language: str) -> SourceFile:
    return SourceFile(
        id="file-id",
        project_id="project-id",
        relative_path=Path(path),
        absolute_path=Path(path),
        language=language,
        content_hash="content-hash",
        size_bytes=0,
    )


def test_code_chunker_keeps_small_symbols_intact_and_ids_stable() -> None:
    source = _source("sample.py", "Python")
    text = "def first():\n    return 1\n\ndef second():\n    return 2\n"
    symbols = ParserDispatcher().parse(source, text).symbols
    chunker = CodeChunker(max_chunk_chars=100)

    chunks = chunker.chunk(source, text, list(symbols))
    repeated = chunker.chunk(source, text, list(symbols))

    assert [chunk.text for chunk in chunks] == [
        "def first():\n    return 1\n",
        "def second():\n    return 2\n",
    ]
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in repeated]
    assert [chunk.span for chunk in chunks] == [SourceSpan(1, 2), SourceSpan(4, 5)]
    assert all(chunk.content_hash == content_hash_for(chunk.text) for chunk in chunks)
    assert all(chunk.symbol_id and chunk.file_id == source.id for chunk in chunks)


def test_oversized_nested_symbol_splits_with_bounded_spans() -> None:
    source = _source("sample.py", "Python")
    text = (
        "class Store:\n"
        "    def read(self):\n"
        "        def inner():\n"
        "            return 'abcdefghijklmnopqrstuvwxyz0123456789'\n"
        "        return inner()\n"
    )
    symbols = ParserDispatcher().parse(source, text).symbols
    chunker = CodeChunker(max_chunk_chars=32)

    chunks = chunker.chunk(source, text, list(symbols))

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 32 for chunk in chunks)
    assert all(1 <= chunk.span.start_line <= chunk.span.end_line <= 5 for chunk in chunks)
    assert any(chunk.metadata.get("oversized_symbol") for chunk in chunks)
    nested_chunk = next(chunk for chunk in chunks if chunk.symbol_name == "Store.read.inner")
    symbols_by_name = {symbol.qualified_name: symbol for symbol in symbols}
    assert nested_chunk.symbol_id == symbols_by_name["Store.read.inner"].id
    assert nested_chunk.metadata["parent_symbol_id"] == symbols_by_name[
        "Store.read.inner"
    ].parent_id
    assert "".join(chunk.text for chunk in chunks) == text
    repeated = chunker.chunk(source, text, list(symbols))
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in repeated]


def test_oversized_symbol_preserves_interior_whitespace_and_comments() -> None:
    source = _source("sample.py", "Python")
    text = (
        "class Store:\n"
        "    def first(self):\n"
        "        return 'first'\n"
        "\n"
        "    def second(self):\n"
        "        return 'second'\n"
        "\n"
        "    # Preserve this note between child symbols.\n"
        "    def third(self):\n"
        "        return 'third'\n"
    )
    symbols = ParserDispatcher().parse(source, text).symbols
    chunker = CodeChunker(max_chunk_chars=64)

    chunks = chunker.chunk(source, text, list(symbols))
    repeated = chunker.chunk(source, text, list(symbols))

    assert len(text) > 64
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(len(chunk.text) <= 64 for chunk in chunks)
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in repeated]
    assert [chunk.text for chunk in chunks] == [chunk.text for chunk in repeated]
    assert any(chunk.text == "\n" and chunk.span == SourceSpan(4, 4) for chunk in chunks)
    assert any(
        chunk.text == "\n    # Preserve this note between child symbols.\n"
        and chunk.span == SourceSpan(7, 8)
        for chunk in chunks
    )

    symbols_by_name = {symbol.qualified_name: symbol for symbol in symbols}
    second_method = next(chunk for chunk in chunks if chunk.symbol_name == "Store.second")
    assert second_method.symbol_id == symbols_by_name["Store.second"].id
    assert second_method.metadata["parent_symbol_id"] == symbols_by_name["Store"].id


def test_top_level_whitespace_between_symbols_does_not_emit_a_chunk() -> None:
    source = _source("sample.py", "Python")
    text = "def first():\n    return 1\n\n\n\ndef second():\n    return 2\n"
    symbols = ParserDispatcher().parse(source, text).symbols

    chunks = CodeChunker(max_chunk_chars=100).chunk(source, text, list(symbols))

    assert [chunk.text for chunk in chunks] == [
        "def first():\n    return 1\n",
        "def second():\n    return 2\n",
    ]


def test_identical_code_fragments_on_one_line_have_distinct_stable_ids() -> None:
    source = _source("module.py", "Python")
    text = "x=" + "a" * 20 + "\n"
    chunker = CodeChunker(max_chunk_chars=2)

    chunks = chunker.chunk(source, text, [])
    repeated = chunker.chunk(source, text, [])
    repeated_fragments = [chunk for chunk in chunks if chunk.text == "aa"]

    assert len(repeated_fragments) > 1
    assert all(chunk.span == SourceSpan(1, 1) for chunk in repeated_fragments)
    assert len({chunk.id for chunk in repeated_fragments}) == len(repeated_fragments)
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in repeated]


def test_large_module_level_text_is_also_bounded() -> None:
    source = _source("module.py", "Python")
    text = "VALUE = 'abcdefghijklmnopqrstuvwxyz'\nOTHER = 'abcdefghijklmnopqrstuvwxyz'\n"

    chunks = CodeChunker(max_chunk_chars=12).chunk(source, text, [])

    assert chunks
    assert all(len(chunk.text) <= 12 for chunk in chunks)
    assert all(chunk.metadata["chunk_kind"] == "module" for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text


def test_prose_chunks_keep_heading_metadata_and_bounded_fallback() -> None:
    source = _source("guide.md", "Markdown")
    text = "# Guide\n\nFirst sentence. Second sentence. Third sentence.\n"
    document = ParserDispatcher().parse(source, text).document
    assert document is not None
    chunker = ProseChunker(max_chunk_chars=20)

    chunks = chunker.chunk(source, document)
    repeated = chunker.chunk(source, document)

    assert all(len(chunk.text) <= 20 for chunk in chunks)
    assert chunks[0].metadata["title"] == "Guide"
    assert all(chunk.metadata["heading_path"] == ["Guide"] for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == (
        "# Guide\nFirst sentence. Second sentence. Third sentence.\n"
    )
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in repeated]
    assert Settings(_env_file=None, max_chunk_chars=20).max_chunk_chars == 20


def test_prose_oversized_paragraph_spans_advance_deterministically() -> None:
    source = _source("notes.txt", "text")
    text = "First sentence.\nSecond sentence is much longer than the configured chunk limit.\n"
    document = ParserDispatcher().parse(source, text).document
    assert document is not None

    chunks = ProseChunker(max_chunk_chars=18).chunk(source, document)

    assert len(chunks) >= 3
    assert all(len(chunk.text) <= 18 for chunk in chunks)
    assert all(chunk.span.start_line <= chunk.span.end_line for chunk in chunks)
    assert chunks[0].span.start_line == chunks[0].span.end_line == 1
    assert all(chunk.span.start_line == chunk.span.end_line == 2 for chunk in chunks[1:])


def test_identical_prose_fragments_on_one_line_have_distinct_stable_ids() -> None:
    source = _source("notes.txt", "text")
    text = "echo echo echo echo\n"
    document = ParserDispatcher().parse(source, text).document
    assert document is not None
    chunker = ProseChunker(max_chunk_chars=5)

    chunks = chunker.chunk(source, document)
    repeated = chunker.chunk(source, document)
    repeated_fragments = [chunk for chunk in chunks if chunk.text == "echo "]

    assert len(repeated_fragments) > 1
    assert all(chunk.span == SourceSpan(1, 1) for chunk in repeated_fragments)
    assert len({chunk.id for chunk in repeated_fragments}) == len(repeated_fragments)
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in repeated]


def test_multiple_paragraphs_and_heading_transitions_stay_separate() -> None:
    source = _source("guide.md", "Markdown")
    text = (
        "# First\n\nFirst paragraph.\n\nSecond paragraph.\n\n"
        "## Child\n\nChild paragraph.\n\n# Next\n\nNext section.\n"
    )
    document = ParserDispatcher().parse(source, text).document
    assert document is not None

    chunks = ProseChunker(max_chunk_chars=100).chunk(source, document)
    paragraphs = [chunk for chunk in chunks if chunk.metadata["document_kind"] == "paragraph"]

    assert [chunk.text for chunk in paragraphs] == [
        "First paragraph.\n",
        "Second paragraph.\n",
        "Child paragraph.\n",
        "Next section.\n",
    ]
    assert [chunk.metadata["heading_path"] for chunk in paragraphs] == [
        ["First"],
        ["First"],
        ["First", "Child"],
        ["Next"],
    ]
