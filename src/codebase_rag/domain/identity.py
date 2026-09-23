"""Deterministic identifiers for parsed symbols and retrieval chunks."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

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
    symbol_id: str | None,
    metadata: dict[str, Any],
) -> str:
    """Return a stable ID from file identity, source range, content, and metadata."""
    identity = {
        "file_id": file_id,
        "span": [span.start_line, span.end_line],
        "content_hash": content_hash_for(text),
        "symbol_id": symbol_id,
        "metadata": metadata,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(f"chunk\0{payload}".encode()).hexdigest()
