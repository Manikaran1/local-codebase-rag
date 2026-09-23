from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest

from codebase_rag.ingestion.hashing import (
    content_hash_bytes,
    content_hash_stream,
    file_id_for,
    project_id_for,
)


def test_content_hash_is_sha256_of_exact_bytes() -> None:
    content = b"repository source\n"

    assert content_hash_bytes(content) == sha256(content).hexdigest()
    assert content_hash_bytes(content) == content_hash_bytes(bytes(content))
    assert content_hash_bytes(content) != content_hash_bytes(content + b"changed")


def test_stream_hash_includes_prefix_and_stops_at_size_limit() -> None:
    content = b"prefix and rest"
    prefix = content[:6]

    result = content_hash_stream(
        BytesIO(content[len(prefix) :]),
        initial_bytes=prefix,
        chunk_size=4,
    )
    assert result == (sha256(content).hexdigest(), len(content))

    assert content_hash_stream(BytesIO(b"12345"), max_bytes=4) is None


def test_project_and_file_ids_are_stable_and_path_sensitive(tmp_path: Path) -> None:
    canonical_root = tmp_path.resolve()
    project_id = project_id_for(canonical_root)

    assert project_id == project_id_for(canonical_root)
    assert len(project_id) == 64
    assert file_id_for(project_id, "src/a.py") == file_id_for(project_id, Path("src/a.py"))
    assert file_id_for(project_id, "src/a.py") != file_id_for(project_id, "src/b.py")
    assert file_id_for(project_id, "src/a.py") != file_id_for(
        project_id_for((tmp_path / "other").resolve()),
        "src/a.py",
    )

    with pytest.raises(ValueError, match="inside the repository"):
        file_id_for(project_id, "../outside.py")
