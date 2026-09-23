"""Markdown and plain-text document parsing with precise line spans."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from codebase_rag.domain.models import DocumentBlock, ParsedDocument, SourceFile, SourceSpan

_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)\s*(?:\r?\n)?$")


class DocumentParser(ABC):
    @abstractmethod
    def supports(self, source: SourceFile) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse(self, source: SourceFile, text: str) -> ParsedDocument:
        raise NotImplementedError


class MarkdownTextParser(DocumentParser):
    """Parse Markdown blocks and text paragraphs without interpreting inline markup."""

    _markdown_extensions = {".md", ".markdown"}

    def supports(self, source: SourceFile) -> bool:
        return source.relative_path.suffix.lower() in self._markdown_extensions | {".txt"}

    def parse(self, source: SourceFile, text: str) -> ParsedDocument:
        if source.relative_path.suffix.lower() in self._markdown_extensions:
            return self.parse_markdown(text)
        return self.parse_plain_text(text)

    def parse_markdown(self, text: str) -> ParsedDocument:
        lines = text.splitlines(keepends=True)
        blocks: list[DocumentBlock] = []
        heading_stack: list[tuple[int, str]] = []
        index = 0

        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue

            fence = _FENCE_OPEN.match(line)
            if fence is not None:
                marker = fence.group(1)
                marker_char = marker[0]
                closing_pattern = (
                    rf"^ {{0,3}}{re.escape(marker_char)}{{{len(marker)},}}"
                    r"\s*(?:\r?\n)?$"
                )
                closing = re.compile(closing_pattern)
                end = index + 1
                while end < len(lines):
                    if closing.match(lines[end]):
                        end += 1
                        break
                    end += 1
                blocks.append(
                    DocumentBlock(
                        kind="code",
                        text="".join(lines[index:end]),
                        span=SourceSpan(index + 1, end),
                        heading_path=tuple(title for _, title in heading_stack),
                    )
                )
                index = end
                continue

            atx_heading = self._atx_heading(line)
            if atx_heading is not None:
                level, title = atx_heading
                self._push_heading(heading_stack, level, title)
                blocks.append(
                    DocumentBlock(
                        kind="heading",
                        text=line,
                        span=SourceSpan(index + 1, index + 1),
                        heading_level=level,
                        heading_path=tuple(item for _, item in heading_stack),
                        title=title,
                    )
                )
                index += 1
                continue

            if index + 1 < len(lines) and _SETEXT.match(lines[index + 1]):
                underline = _SETEXT.match(lines[index + 1])
                assert underline is not None
                level = 1 if underline.group(1).startswith("=") else 2
                title = line.strip()
                self._push_heading(heading_stack, level, title)
                blocks.append(
                    DocumentBlock(
                        kind="heading",
                        text=line + lines[index + 1],
                        span=SourceSpan(index + 1, index + 2),
                        heading_level=level,
                        heading_path=tuple(item for _, item in heading_stack),
                        title=title,
                    )
                )
                index += 2
                continue

            end = index + 1
            while end < len(lines):
                if not lines[end].strip() or self._atx_heading(lines[end]) is not None:
                    break
                if _FENCE_OPEN.match(lines[end]) is not None:
                    break
                if end + 1 < len(lines) and _SETEXT.match(lines[end + 1]):
                    break
                end += 1
            blocks.append(
                DocumentBlock(
                    kind="paragraph",
                    text="".join(lines[index:end]),
                    span=SourceSpan(index + 1, end),
                    heading_path=tuple(item for _, item in heading_stack),
                )
            )
            index = end

        return ParsedDocument(blocks=tuple(blocks))

    def parse_plain_text(self, text: str, *, is_fallback: bool = False) -> ParsedDocument:
        lines = text.splitlines(keepends=True)
        blocks: list[DocumentBlock] = []
        index = 0
        while index < len(lines):
            if not lines[index].strip():
                index += 1
                continue
            end = index + 1
            while end < len(lines) and lines[end].strip():
                end += 1
            blocks.append(
                DocumentBlock(
                    kind="paragraph",
                    text="".join(lines[index:end]),
                    span=SourceSpan(index + 1, end),
                )
            )
            index = end
        return ParsedDocument(blocks=tuple(blocks), is_fallback=is_fallback)

    @staticmethod
    def _atx_heading(line: str) -> tuple[int, str] | None:
        content = line.rstrip("\r\n")
        match = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)|[ \t]*)$", content)
        if match is None:
            return None
        title = match.group(2) or ""
        title = re.sub(r"[ \t]+#+[ \t]*$", "", title).strip()
        return len(match.group(1)), title

    @staticmethod
    def _push_heading(stack: list[tuple[int, str]], level: int, title: str) -> None:
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
