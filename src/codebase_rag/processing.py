"""Source-to-chunk application boundary for local repository indexing."""

from __future__ import annotations

from dataclasses import dataclass

from codebase_rag.chunking.code import CodeChunker
from codebase_rag.chunking.prose import ProseChunker
from codebase_rag.config.settings import Settings
from codebase_rag.domain.models import Chunk, ParsedDocument, SourceFile
from codebase_rag.parsing.dispatcher import ParserDispatcher, ParseResult


@dataclass(frozen=True)
class SourceProcessingResult:
    parse_result: ParseResult
    chunks: tuple[Chunk, ...]


class SourceProcessor:
    """Parse one scanned source and return deterministic domain chunks."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        parser: ParserDispatcher | None = None,
    ) -> None:
        resolved_settings = settings if settings is not None else Settings()
        self._parser = parser if parser is not None else ParserDispatcher()
        self._code_chunker = CodeChunker(resolved_settings.max_chunk_chars)
        self._prose_chunker = ProseChunker(resolved_settings.max_chunk_chars)

    def process(self, source: SourceFile, text: str) -> SourceProcessingResult:
        parsed = self._parser.parse(source, text)
        if parsed.kind == "python":
            chunks = self._code_chunker.chunk(source, text, list(parsed.symbols))
        else:
            document = parsed.document if parsed.document is not None else ParsedDocument(())
            chunks = self._prose_chunker.chunk(source, document)
        return SourceProcessingResult(parse_result=parsed, chunks=tuple(chunks))
