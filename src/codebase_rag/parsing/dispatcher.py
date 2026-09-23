"""Choose a parser while keeping implementation-specific data private."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from codebase_rag.domain.models import ParsedDocument, SourceFile, Symbol
from codebase_rag.parsing.documents import MarkdownTextParser
from codebase_rag.parsing.python_parser import PythonParser

ParseKind = Literal["python", "document", "fallback"]


@dataclass(frozen=True)
class ParseResult:
    kind: ParseKind
    symbols: tuple[Symbol, ...] = ()
    document: ParsedDocument | None = None
    diagnostics: tuple[str, ...] = ()


class ParserDispatcher:
    """Dispatch Python and document sources, with explicit plain-text fallback."""

    def __init__(self) -> None:
        self._python_parser = PythonParser()
        self._document_parser = MarkdownTextParser()

    def parse(self, source: SourceFile, text: str) -> ParseResult:
        if self._python_parser.supports(source):
            try:
                symbols, diagnostics = self._python_parser.parse_with_diagnostics(source, text)
            except UnicodeEncodeError:
                document = self._document_parser.parse_plain_text(text, is_fallback=True)
                return ParseResult(
                    kind="fallback",
                    document=document,
                    diagnostics=(
                        "Python source was not valid UTF-8 text; used plain-text fallback.",
                    ),
                )
            return ParseResult(kind="python", symbols=tuple(symbols), diagnostics=diagnostics)

        if self._document_parser.supports(source):
            return ParseResult(kind="document", document=self._document_parser.parse(source, text))

        document = self._document_parser.parse_plain_text(text, is_fallback=True)
        language = source.language or source.relative_path.suffix or "source"
        return ParseResult(
            kind="fallback",
            document=document,
            diagnostics=(
                f"No structural parser for {language}; used plain-text fallback.",
            ),
        )
