"""Core domain contracts used across the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceSpan:
    start_line: int
    end_line: int


@dataclass(frozen=True)
class SourceFile:
    id: str
    project_id: str
    relative_path: Path
    absolute_path: Path
    language: str | None
    content_hash: str
    size_bytes: int


@dataclass(frozen=True)
class Symbol:
    id: str
    file_id: str
    name: str
    qualified_name: str
    kind: str
    language: str
    span: SourceSpan
    parent_id: str | None = None
    signature: str | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    project_id: str
    file_id: str
    relative_path: Path
    text: str
    span: SourceSpan
    language: str | None = None
    symbol_id: str | None = None
    symbol_name: str | None = None
    symbol_kind: str | None = None
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    rank: int
    fusion_score: float
    vector_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class Citation:
    relative_path: Path
    start_line: int
    end_line: int
    chunk_id: str
    symbol_name: str | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
