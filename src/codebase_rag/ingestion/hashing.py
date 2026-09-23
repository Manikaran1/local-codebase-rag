"""Stable identities and content hashing for repository sources."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import BinaryIO


def content_hash_bytes(content: bytes) -> str:
    """Return the SHA-256 digest of a file's exact byte contents."""
    return sha256(content).hexdigest()


def content_hash_stream(
    stream: BinaryIO,
    *,
    initial_bytes: bytes = b"",
    max_bytes: int | None = None,
    chunk_size: int = 64 * 1024,
) -> tuple[str, int] | None:
    """Hash a binary stream in bounded memory, returning None past a size limit."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes cannot be negative")
    if max_bytes is not None and len(initial_bytes) > max_bytes:
        return None

    digest = sha256(initial_bytes)
    size_bytes = len(initial_bytes)
    while True:
        read_size = chunk_size
        if max_bytes is not None:
            read_size = min(read_size, max_bytes - size_bytes + 1)
        chunk = stream.read(read_size)
        if not chunk:
            break
        size_bytes += len(chunk)
        if max_bytes is not None and size_bytes > max_bytes:
            return None
        digest.update(chunk)

    return digest.hexdigest(), size_bytes


def project_id_for(canonical_repository_path: Path) -> str:
    """Return a stable project ID for an absolute, resolved repository path."""
    if not canonical_repository_path.is_absolute():
        raise ValueError("canonical_repository_path must be absolute")

    payload = f"project\0{canonical_repository_path.as_posix()}".encode()
    return sha256(payload).hexdigest()


def file_id_for(project_id: str, relative_path: str | Path) -> str:
    """Return a stable file ID for a project and repository-relative POSIX path."""
    path_text = relative_path.as_posix() if isinstance(relative_path, Path) else relative_path
    normalized_path = PurePosixPath(path_text.replace("\\", "/"))
    if normalized_path.is_absolute() or not normalized_path.parts or ".." in normalized_path.parts:
        raise ValueError("relative_path must remain inside the repository")

    payload = f"file\0{project_id}\0{normalized_path.as_posix()}".encode()
    return sha256(payload).hexdigest()
