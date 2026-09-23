"""Bounded prose chunks that preserve document headings and source lines."""

from __future__ import annotations

from typing import Any

from codebase_rag.chunking.base import DocumentChunker
from codebase_rag.chunking.splitting import SourceLineMap, split_offsets
from codebase_rag.domain.identity import chunk_id_for, content_hash_for
from codebase_rag.domain.models import Chunk, ParsedDocument, SourceFile, SourceSpan


class ProseChunker(DocumentChunker):
    """Keep document blocks intact where possible and split long blocks safely."""

    def __init__(self, max_chunk_chars: int = 4000) -> None:
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be greater than zero")
        self._max_chunk_chars = max_chunk_chars

    def chunk(self, source: SourceFile, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for block in document.blocks:
            metadata: dict[str, Any] = {
                "chunk_kind": "prose",
                "document_kind": block.kind,
                "heading_path": list(block.heading_path),
                "parser_fallback": document.is_fallback,
            }
            if block.heading_level is not None:
                metadata["heading_level"] = block.heading_level
            if block.title is not None:
                metadata["title"] = block.title

            offsets = split_offsets(
                block.text,
                self._max_chunk_chars,
                prefer_sentences=block.kind == "paragraph",
            )
            line_map = SourceLineMap.from_text(block.text)
            for start, end in offsets:
                text = block.text[start:end]
                if not text:
                    continue
                span = line_map.span(block.span.start_line, start, end)
                chunks.append(self._make_chunk(source, text, span, metadata))
        return chunks

    @staticmethod
    def _make_chunk(
        source: SourceFile,
        text: str,
        span: SourceSpan,
        metadata: dict[str, Any],
    ) -> Chunk:
        return Chunk(
            id=chunk_id_for(
                file_id=source.id,
                span=span,
                text=text,
                symbol_id=None,
                metadata=metadata,
            ),
            project_id=source.project_id,
            file_id=source.id,
            relative_path=source.relative_path,
            text=text,
            span=span,
            language=source.language,
            content_hash=content_hash_for(text),
            metadata=dict(metadata),
        )
