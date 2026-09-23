"""Deterministic identifiers for parsed symbols and retrieval chunks."""

from __future__ import annotations

import json
from hashlib import sha256

from codebase_rag.domain.models import SourceSpan


def content_hash_for(text: str) -> str:
    """Return the SHA-256 digest of a chunk's exact UTF-8 text."""
    return sha256(text.encode("utf-8")).hexdigest()


def symbol_id_for(
    file_id: str,
    qualified_name: str,
    kind: str,
    span: SourceSpan,
) -> str:
    """Return a stable ID for a symbol in a source file."""
    payload = f"symbol\0{file_id}\0{qualified_name}\0{kind}\0{span.start_line}\0{span.end_line}"
    return sha256(payload.encode("utf-8")).hexdigest()


def chunk_id_for(
    *,
    file_id: str,
    span: SourceSpan,
    text: str,
    source_offset: int | None = None,
) -> str:
    """Return a stable ID from source identity, content, and optional source offset.

    The offset disambiguates identical chunks that share a line-only source span.
    """
    identity: list[str | int] = [
        file_id,
        span.start_line,
        span.end_line,
        content_hash_for(text),
    ]
    if source_offset is not None:
        identity.append(source_offset)
    payload = json.dumps(identity, separators=(",", ":"), ensure_ascii=False)
    return sha256(f"chunk\0{payload}".encode()).hexdigest()
