from pathlib import Path

from tree_sitter import Language, Node, Parser, Tree

from codebase_rag.domain.models import SourceFile, SourceSpan, Symbol
from codebase_rag.parsing.dispatcher import ParserDispatcher
from codebase_rag.parsing.python_parser import PythonParser


def _source(path: str = "sample.py", language: str = "Python") -> SourceFile:
    return SourceFile(
        id="file-id",
        project_id="project-id",
        relative_path=Path(path),
        absolute_path=Path(path),
        language=language,
        content_hash="content-hash",
        size_bytes=0,
    )


def test_extracts_nested_and_decorated_symbols_with_exact_spans() -> None:
    source = _source("examples/sample_repo/structure.py")
    text = source.absolute_path.read_text(encoding="utf-8")

    symbols = PythonParser().parse(source, text)
    by_name = {symbol.qualified_name: symbol for symbol in symbols}

    assert set(by_name) == {
        "normalize",
        "Repository",
        "Repository.load",
        "Repository.load.clean",
        "Repository.Cache",
        "Repository.Cache.get",
        "refresh",
    }
    assert by_name["normalize"].kind == "function"
    assert by_name["normalize"].span == SourceSpan(5, 6)
    assert by_name["Repository"].kind == "class"
    assert by_name["Repository"].span == SourceSpan(9, 18)
    assert by_name["Repository.load"].kind == "method"
    assert by_name["Repository.load"].span == SourceSpan(10, 14)
    assert by_name["Repository.load.clean"].kind == "function"
    assert by_name["Repository.load.clean"].span == SourceSpan(11, 12)
    assert by_name["Repository.load.clean"].parent_id == by_name["Repository.load"].id
    assert by_name["Repository.Cache"].parent_id == by_name["Repository"].id
    assert by_name["Repository.Cache.get"].parent_id == by_name["Repository.Cache"].id
    assert by_name["Repository.load"].signature == "def load(self, value: str) -> str:"
    assert by_name["refresh"].span == SourceSpan(21, 23)
    assert by_name["refresh"].signature == "async def refresh(limit: int = 1) -> None:"


def test_symbol_ids_are_deterministic() -> None:
    parser = PythonParser()
    source = _source()
    text = "class Example:\n    def method(self) -> int:\n        return 1\n"

    first = parser.parse(source, text)
    second = parser.parse(source, text)

    assert [symbol.id for symbol in first] == [symbol.id for symbol in second]
    assert all(len(symbol.id) == 64 for symbol in first)


def test_malformed_source_keeps_recoverable_siblings() -> None:
    source = _source()
    text = "def broken(:\n    pass\n\ndef good(value: str) -> str:\n    return value\n"

    result = ParserDispatcher().parse(source, text)

    assert result.kind == "python"
    assert result.diagnostics
    assert {symbol.name for symbol in result.symbols} == {"good"}
    assert all(isinstance(symbol, Symbol) for symbol in result.symbols)


def test_tree_sitter_objects_do_not_escape_parse_results() -> None:
    source = _source()
    result = ParserDispatcher().parse(
        source,
        "class Example:\n    def method(self) -> int:\n        return 1\n",
    )

    assert result.kind == "python"
    for symbol in result.symbols:
        for value in vars(symbol).values():
            assert not isinstance(value, (Language, Node, Parser, Tree))


def test_unsupported_language_uses_explicit_plain_text_fallback() -> None:
    result = ParserDispatcher().parse(_source("source.js", "JavaScript"), "const value = 1;\n")

    assert result.kind == "fallback"
    assert result.symbols == ()
    assert result.document is not None
    assert result.document.is_fallback
    assert result.document.blocks[0].text == "const value = 1;\n"
    assert result.diagnostics
