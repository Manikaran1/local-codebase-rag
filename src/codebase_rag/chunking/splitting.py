"""Shared lossless bounded text splitting for code and prose chunkers."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from codebase_rag.domain.models import SourceSpan

_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def split_offsets(
    text: str,
    max_chars: int,
    *,
    prefer_sentences: bool = False,
) -> list[tuple[int, int]]:
    """Split text without loss, preferring source lines and then prose boundaries."""
    if not text:
        return []
    pieces: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        limit = min(cursor + max_chars, len(text))
        end = limit
        if limit < len(text):
            line_ends = [match.end() for match in _LINE_BREAK.finditer(text, cursor, limit)]
            if line_ends:
                end = line_ends[-1]
            elif prefer_sentences:
                sentence_ends = [
                    match.end()
                    for match in re.finditer(r"(?<=[.!?])[\"')\]]?\s+", text[cursor:limit])
                ]
                if sentence_ends:
                    end = cursor + sentence_ends[-1]
                else:
                    whitespace_ends = [
                        match.end()
                        for match in re.finditer(r"\s+", text[cursor:limit])
                    ]
                    if whitespace_ends:
                        end = cursor + whitespace_ends[-1]
        if end <= cursor:
            end = limit
        pieces.append((cursor, end))
        cursor = end
    return pieces


@dataclass(frozen=True)
class SourceLineMap:
    """Map text offsets to line indices using one scan and logarithmic lookups."""

    line_starts: tuple[int, ...]

    @classmethod
    def from_text(cls, text: str) -> SourceLineMap:
        starts = [0]
        starts.extend(match.end() for match in _LINE_BREAK.finditer(text))
        return cls(tuple(starts))

    def span(self, start_line: int, start: int, end: int) -> SourceSpan:
        """Map a half-open text range to one-based inclusive source lines."""
        last_character_offset = max(start, end - 1)
        first_line_offset = max(0, bisect_right(self.line_starts, start) - 1)
        last_line_offset = max(0, bisect_right(self.line_starts, last_character_offset) - 1)
        return SourceSpan(
            start_line=start_line + first_line_offset,
            end_line=start_line + last_line_offset,
        )
