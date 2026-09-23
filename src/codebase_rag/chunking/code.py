"""Structure-first code chunking with bounded deterministic fallbacks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from codebase_rag.chunking.base import Chunker
from codebase_rag.chunking.splitting import SourceLineMap, split_offsets
from codebase_rag.domain.identity import chunk_id_for, content_hash_for
from codebase_rag.domain.models import Chunk, SourceFile, SourceSpan, Symbol


class CodeChunker(Chunker):
    """Keep complete symbols together until the configured size limit is reached."""

    def __init__(self, max_chunk_chars: int = 4000) -> None:
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be greater than zero")
        self._max_chunk_chars = max_chunk_chars

    def chunk(self, source: SourceFile, text: str, symbols: list[Symbol]) -> list[Chunk]:
        lines = text.splitlines(keepends=True)
        if not lines:
            return []
        line_offsets: list[int] = []
        source_offset = 0
        for line in lines:
            line_offsets.append(source_offset)
            source_offset += len(line)
        children_by_parent: dict[str, list[Symbol]] = defaultdict(list)
        roots: list[Symbol] = []
        for symbol in symbols:
            if symbol.file_id != source.id:
                continue
            if symbol.parent_id is None:
                roots.append(symbol)
            else:
                children_by_parent[symbol.parent_id].append(symbol)

        roots.sort(key=lambda item: (item.span.start_line, item.span.end_line, item.qualified_name))
        for children in children_by_parent.values():
            children.sort(
                key=lambda item: (item.span.start_line, item.span.end_line, item.qualified_name)
            )

        result: list[Chunk] = []
        cursor = 1
        for group in self._overlapping_groups(roots):
            start_line = max(1, min(symbol.span.start_line for symbol in group))
            end_line = min(len(lines), max(symbol.span.end_line for symbol in group))
            if start_line > end_line:
                continue
            if cursor < start_line:
                self._emit_raw(
                    source,
                    lines,
                    line_offsets,
                    cursor,
                    start_line - 1,
                    result,
                    metadata={"chunk_kind": "module"},
                )
            self._emit_symbol_group(
                source,
                lines,
                line_offsets,
                group,
                children_by_parent,
                result,
                None,
            )
            cursor = max(cursor, end_line + 1)

        if cursor <= len(lines):
            self._emit_raw(
                source,
                lines,
                line_offsets,
                cursor,
                len(lines),
                result,
                metadata={"chunk_kind": "module"},
            )
        return result

    def _emit_symbol_group(
        self,
        source: SourceFile,
        lines: list[str],
        line_offsets: list[int],
        group: list[Symbol],
        children_by_parent: dict[str, list[Symbol]],
        result: list[Chunk],
        parent: Symbol | None,
    ) -> None:
        start_line = max(1, min(symbol.span.start_line for symbol in group))
        end_line = min(len(lines), max(symbol.span.end_line for symbol in group))
        if start_line > end_line:
            return
        text = self._line_slice(lines, start_line, end_line)

        if len(group) > 1:
            metadata: dict[str, Any] = {
                "chunk_kind": "structural",
                "symbol_ids": [symbol.id for symbol in group],
                "symbol_names": [symbol.qualified_name for symbol in group],
                "symbol_kinds": [symbol.kind for symbol in group],
            }
            if parent is not None:
                metadata["parent_symbol_id"] = parent.id
            self._emit_bounded(
                source,
                text,
                start_line,
                line_offsets[start_line - 1],
                group[0],
                metadata,
                result,
                prefer_sentences=False,
            )
            return

        symbol = group[0]
        direct_children = [
            child
            for child in children_by_parent.get(symbol.id, [])
            if child.span.start_line >= start_line and child.span.end_line <= end_line
        ]
        if len(text) <= self._max_chunk_chars or not direct_children:
            metadata = {
                "chunk_kind": (
                    "structural" if len(text) <= self._max_chunk_chars else "symbol_fragment"
                ),
                "symbol_qualified_name": symbol.qualified_name,
            }
            if len(text) <= self._max_chunk_chars:
                descendants = self._descendants(symbol, children_by_parent)
                if descendants:
                    metadata["contained_symbols"] = [item.qualified_name for item in descendants]
            else:
                metadata["oversized_symbol"] = True
            if parent is not None:
                metadata["parent_symbol_id"] = parent.id
            if len(text) <= self._max_chunk_chars:
                result.append(
                    self._make_chunk(
                        source,
                        text,
                        SourceSpan(start_line, end_line),
                        line_offsets[start_line - 1],
                        symbol,
                        metadata,
                    )
                )
            else:
                self._emit_bounded(
                    source,
                    text,
                    start_line,
                    line_offsets[start_line - 1],
                    symbol,
                    metadata,
                    result,
                )
            return

        cursor = start_line
        for child_group in self._overlapping_groups(direct_children):
            child_start = max(start_line, min(item.span.start_line for item in child_group))
            child_end = min(end_line, max(item.span.end_line for item in child_group))
            if cursor < child_start:
                self._emit_raw(
                    source,
                    lines,
                    line_offsets,
                    cursor,
                    child_start - 1,
                    result,
                    symbol=symbol,
                    metadata={"chunk_kind": "symbol_fragment", "oversized_symbol": True},
                    preserve_whitespace=True,
                )
            self._emit_symbol_group(
                source,
                lines,
                line_offsets,
                child_group,
                children_by_parent,
                result,
                symbol,
            )
            cursor = max(cursor, child_end + 1)
        if cursor <= end_line:
            self._emit_raw(
                source,
                lines,
                line_offsets,
                cursor,
                end_line,
                result,
                symbol=symbol,
                metadata={"chunk_kind": "symbol_fragment", "oversized_symbol": True},
                preserve_whitespace=True,
            )

    def _emit_raw(
        self,
        source: SourceFile,
        lines: list[str],
        line_offsets: list[int],
        start_line: int,
        end_line: int,
        result: list[Chunk],
        *,
        symbol: Symbol | None = None,
        metadata: dict[str, Any],
        preserve_whitespace: bool = False,
    ) -> None:
        text = self._line_slice(lines, start_line, end_line)
        if not preserve_whitespace and not text.strip():
            return
        complete_metadata = dict(metadata)
        if symbol is not None:
            complete_metadata["symbol_qualified_name"] = symbol.qualified_name
        self._emit_bounded(
            source,
            text,
            start_line,
            line_offsets[start_line - 1],
            symbol,
            complete_metadata,
            result,
        )

    def _emit_bounded(
        self,
        source: SourceFile,
        text: str,
        start_line: int,
        source_offset: int,
        symbol: Symbol | None,
        metadata: dict[str, Any],
        result: list[Chunk],
        *,
        prefer_sentences: bool = False,
    ) -> None:
        line_map = SourceLineMap.from_text(text)
        for start, end in split_offsets(
            text,
            self._max_chunk_chars,
            prefer_sentences=prefer_sentences,
        ):
            piece = text[start:end]
            if not piece:
                continue
            span = line_map.span(start_line, start, end)
            result.append(
                self._make_chunk(
                    source,
                    piece,
                    span,
                    source_offset + start,
                    symbol,
                    metadata,
                )
            )

    def _make_chunk(
        self,
        source: SourceFile,
        text: str,
        span: SourceSpan,
        source_offset: int,
        symbol: Symbol | None,
        metadata: dict[str, Any],
    ) -> Chunk:
        return Chunk(
            id=chunk_id_for(
                file_id=source.id,
                span=span,
                text=text,
                source_offset=source_offset,
            ),
            project_id=source.project_id,
            file_id=source.id,
            relative_path=source.relative_path,
            text=text,
            span=span,
            language=source.language,
            symbol_id=symbol.id if symbol is not None else None,
            symbol_name=symbol.qualified_name if symbol is not None else None,
            symbol_kind=symbol.kind if symbol is not None else None,
            content_hash=content_hash_for(text),
            metadata=dict(metadata),
        )

    def _descendants(
        self,
        symbol: Symbol,
        children_by_parent: dict[str, list[Symbol]],
    ) -> list[Symbol]:
        descendants: list[Symbol] = []
        pending = list(children_by_parent.get(symbol.id, []))
        while pending:
            child = pending.pop(0)
            descendants.append(child)
            pending[0:0] = children_by_parent.get(child.id, [])
        return descendants

    @staticmethod
    def _line_slice(lines: list[str], start_line: int, end_line: int) -> str:
        return "".join(lines[start_line - 1 : end_line])

    @staticmethod
    def _overlapping_groups(symbols: Iterable[Symbol]) -> list[list[Symbol]]:
        groups: list[list[Symbol]] = []
        for symbol in symbols:
            if not groups or symbol.span.start_line > max(
                item.span.end_line for item in groups[-1]
            ):
                groups.append([symbol])
            else:
                groups[-1].append(symbol)
        return groups
