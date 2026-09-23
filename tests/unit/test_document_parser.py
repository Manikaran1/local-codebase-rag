from pathlib import Path

from codebase_rag.domain.models import SourceFile, SourceSpan
from codebase_rag.parsing.dispatcher import ParserDispatcher
from codebase_rag.parsing.documents import MarkdownTextParser


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


def test_markdown_blocks_keep_heading_path_and_code_fences() -> None:
    source = _source("guide.md", "Markdown")
    text = (
        "# Guide\n\nIntro paragraph.\n\n## Details\nBody text.\n\n"
        "```python\n# not a heading\nvalue = 1\n```\n\n### Safety\nSafe parsing.\n"
    )

    document = MarkdownTextParser().parse(source, text)

    assert [block.kind for block in document.blocks] == [
        "heading",
        "paragraph",
        "heading",
        "paragraph",
        "code",
        "heading",
        "paragraph",
    ]
    assert document.blocks[0].span == SourceSpan(1, 1)
    assert document.blocks[1].span == SourceSpan(3, 3)
    assert document.blocks[3].heading_path == ("Guide", "Details")
    assert document.blocks[4].text == "```python\n# not a heading\nvalue = 1\n```\n"
    assert document.blocks[4].heading_path == ("Guide", "Details")
    assert document.blocks[5].heading_path == ("Guide", "Details", "Safety")
    assert document == MarkdownTextParser().parse(source, text)


def test_markdown_recognizes_setext_headings() -> None:
    document = MarkdownTextParser().parse_markdown("Overview\n========\n\nDetails\n-------\n")

    assert [(block.title, block.heading_level, block.span) for block in document.blocks] == [
        ("Overview", 1, SourceSpan(1, 2)),
        ("Details", 2, SourceSpan(4, 5)),
    ]


def test_plain_text_splits_on_blank_lines_and_empty_is_safe() -> None:
    parser = MarkdownTextParser()
    document = parser.parse(_source("notes.txt", "text"), "First paragraph.\n\nSecond.\n")

    assert [block.text for block in document.blocks] == ["First paragraph.\n", "Second.\n"]
    assert [block.span for block in document.blocks] == [SourceSpan(1, 1), SourceSpan(3, 3)]
    assert parser.parse_plain_text("").blocks == ()


def test_dispatcher_uses_markdown_for_supported_document_extensions() -> None:
    result = ParserDispatcher().parse(_source("guide.markdown", "Markdown"), "# Guide\n")

    assert result.kind == "document"
    assert result.document is not None
    assert result.document.blocks[0].title == "Guide"
