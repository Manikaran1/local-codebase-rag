from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from codebase_rag.config.settings import DEFAULT_SUPPORTED_EXTENSIONS, Settings
from codebase_rag.domain.models import SourceFile
from codebase_rag.ingestion.filtering import RepositoryConfigurationError
from codebase_rag.ingestion.hashing import file_id_for, project_id_for
from codebase_rag.ingestion.repository_scanner import RepositoryScanner


def test_scan_returns_sorted_supported_files_with_complete_metadata(tmp_path: Path) -> None:
    repository = tmp_path / "project"
    contents = {
        "z.py": b"same bytes\n",
        "nested/a.py": b"same bytes\n",
        "changed.py": b"different bytes\n",
        "README.md": b"# Notes\n",
        "notes.txt": "naïve text\n".encode(),
        "unsupported.log": b"not an indexable extension\n",
    }
    for relative_path, content in contents.items():
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    scanner = RepositoryScanner(Settings(_env_file=None))
    first_scan = scanner.scan(repository)
    second_scan = scanner.scan(repository)
    canonical_root = repository.resolve()
    project_id = project_id_for(canonical_root)

    expected_paths = [
        "README.md",
        "changed.py",
        "nested/a.py",
        "notes.txt",
        "z.py",
    ]
    assert [source.relative_path.as_posix() for source in first_scan] == expected_paths
    assert [source.id for source in first_scan] == [source.id for source in second_scan]
    assert [source.project_id for source in first_scan] == [project_id] * len(first_scan)
    assert len({source.id for source in first_scan}) == len(first_scan)

    by_relative_path = {source.relative_path.as_posix(): source for source in first_scan}
    assert by_relative_path["README.md"].language == "Markdown"
    assert by_relative_path["changed.py"].language == "Python"
    assert by_relative_path["notes.txt"].language == "text"
    assert by_relative_path["z.py"].content_hash == by_relative_path["nested/a.py"].content_hash
    assert by_relative_path["changed.py"].content_hash != by_relative_path["z.py"].content_hash

    for relative_path, content in contents.items():
        source = by_relative_path.get(relative_path)
        if source is None:
            continue
        assert isinstance(source, SourceFile)
        assert source.id == file_id_for(project_id, relative_path)
        assert source.relative_path == Path(relative_path)
        assert source.absolute_path == canonical_root / relative_path
        assert source.absolute_path.is_absolute()
        assert source.content_hash == sha256(content).hexdigest()
        assert source.size_bytes == len(content)


def test_project_identity_uses_canonical_repository_path(tmp_path: Path) -> None:
    repository = tmp_path / "project"
    repository.mkdir()
    (repository / "source.py").write_text("x = 1\n", encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(repository, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are not available")

    scan = RepositoryScanner(Settings(_env_file=None))
    direct_project_id = scan.scan(repository)[0].project_id
    alias_project_id = scan.scan(alias)[0].project_id

    assert direct_project_id == alias_project_id


def test_repository_ignore_rules_and_nested_negation_are_respected(tmp_path: Path) -> None:
    repository = tmp_path / "project"
    repository.mkdir()
    (repository / ".gitignore").write_text(
        "ignored.py\nignored_dir/\n!ignored_dir/keep.py\nnested/skip.py\n",
        encoding="utf-8",
    )
    (repository / "included.py").write_text("x = 1\n", encoding="utf-8")
    (repository / "ignored.py").write_text("x = 2\n", encoding="utf-8")
    (repository / "ignored_dir" / "hidden.py").parent.mkdir(parents=True)
    (repository / "ignored_dir" / "hidden.py").write_text("x = 3\n", encoding="utf-8")
    (repository / "ignored_dir" / "keep.py").write_text("x = 4\n", encoding="utf-8")
    (repository / "nested" / ".gitignore").parent.mkdir(parents=True)
    (repository / "nested" / ".gitignore").write_text(
        "local-ignore.py\n!skip.py\n",
        encoding="utf-8",
    )
    (repository / "nested" / "skip.py").write_text("x = 5\n", encoding="utf-8")
    (repository / "nested" / "local-ignore.py").write_text("x = 6\n", encoding="utf-8")
    (repository / "nested" / "keep.py").write_text("x = 7\n", encoding="utf-8")

    sources = RepositoryScanner(Settings(_env_file=None)).scan(repository)

    assert [source.relative_path.as_posix() for source in sources] == [
        "ignored_dir/keep.py",
        "included.py",
        "nested/keep.py",
        "nested/skip.py",
    ]


def test_default_exclusions_prune_common_generated_directories(tmp_path: Path) -> None:
    repository = tmp_path / "project"
    repository.mkdir()
    (repository / "keep.py").write_text("x = 1\n", encoding="utf-8")
    excluded_directories = (
        ".git",
        ".pytest_cache",
        "__pycache__",
        ".codebase-rag",
        "build",
        "dist",
        "indexes",
        "models",
        "node_modules",
        "vector_store",
        "venv",
        ".venv",
        ".vscode",
    )
    for directory in excluded_directories:
        path = repository / directory / "hidden.py"
        path.parent.mkdir(parents=True)
        path.write_text("x = 2\n", encoding="utf-8")

    (repository / "state.db").write_text("local state", encoding="utf-8")
    (repository / "editor.swp").write_text("editor scratch", encoding="utf-8")
    (repository / "cache.pyc").write_text("compiled bytecode", encoding="utf-8")
    (repository / "Thumbs.db").write_text("metadata", encoding="utf-8")
    extensions = {
        **DEFAULT_SUPPORTED_EXTENSIONS,
        ".db": "text",
        ".swp": "text",
        ".pyc": "Python bytecode",
    }
    settings = Settings(_env_file=None, supported_extensions=extensions)

    sources = RepositoryScanner(settings).scan(repository)

    assert [source.relative_path.as_posix() for source in sources] == ["keep.py"]


def test_custom_default_exclusions_are_injectable(tmp_path: Path) -> None:
    repository = tmp_path / "project"
    (repository / "custom-cache").mkdir(parents=True)
    (repository / "custom-cache" / "hidden.py").write_text("x = 2\n", encoding="utf-8")
    (repository / "keep.py").write_text("x = 1\n", encoding="utf-8")
    settings = Settings(_env_file=None, default_ignore_patterns=("custom-cache/",))

    sources = RepositoryScanner(settings).scan(repository)

    assert [source.relative_path.as_posix() for source in sources] == ["keep.py"]


def test_default_exclusion_negation_can_restore_a_child_file(tmp_path: Path) -> None:
    repository = tmp_path / "project"
    (repository / "cache").mkdir(parents=True)
    (repository / "cache" / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (repository / "cache" / "skip.py").write_text("x = 2\n", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        default_ignore_patterns=("cache/", "!cache/keep.py"),
    )

    sources = RepositoryScanner(settings).scan(repository)

    assert [source.relative_path.as_posix() for source in sources] == ["cache/keep.py"]


def test_configurable_file_size_limit_accepts_boundary_and_skips_oversized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "project"
    repository.mkdir()
    (repository / "below.txt").write_bytes(b"1234")
    (repository / "boundary.txt").write_bytes(b"12345")
    oversized = repository / "over.txt"
    oversized.write_bytes(b"123456")
    settings = Settings(_env_file=None, max_file_size_bytes=5)
    original_open = Path.open
    opened_paths: list[Path] = []

    def track_opens(path: Path, *args: object, **kwargs: object) -> object:
        opened_paths.append(path)
        if path == oversized:
            raise AssertionError("oversized files must be checked before opening")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_opens)

    sources = RepositoryScanner(settings).scan(repository)

    assert [source.relative_path.as_posix() for source in sources] == [
        "below.txt",
        "boundary.txt",
    ]
    assert [source.size_bytes for source in sources] == [4, 5]
    assert oversized not in opened_paths


def test_binary_files_are_skipped_without_rejecting_text_files(tmp_path: Path) -> None:
    repository = tmp_path / "project"
    repository.mkdir()
    (repository / "text.py").write_bytes(b"print('hello')\n")
    (repository / "nul.py").write_bytes(b"print('hello')\0\xff")
    (repository / "controls.md").write_bytes(b"\x01\x02\x03\x04\x05\x06\x07")

    sources = RepositoryScanner(Settings(_env_file=None)).scan(repository)

    assert [source.relative_path.as_posix() for source in sources] == ["text.py"]


def test_unknown_extension_is_not_indexed_and_custom_extensions_are_injectable(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    repository.mkdir()
    (repository / "note.unknown").write_text("plain text", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        supported_extensions={".unknown": "custom"},
    )

    sources = RepositoryScanner(settings).scan(repository)

    assert [(source.relative_path.as_posix(), source.language) for source in sources] == [
        ("note.unknown", "custom")
    ]


def test_invalid_repository_roots_raise_meaningful_errors(tmp_path: Path) -> None:
    scanner = RepositoryScanner(Settings(_env_file=None))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        scanner.scan(tmp_path / "missing")

    file_path = tmp_path / "file.py"
    file_path.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        scanner.scan(file_path)


def test_unreadable_source_file_is_skipped_without_failing_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "project"
    repository.mkdir()
    readable = repository / "readable.py"
    unreadable = repository / "unreadable.py"
    readable.write_text("x = 1\n", encoding="utf-8")
    unreadable.write_text("x = 2\n", encoding="utf-8")
    original_open = Path.open

    def open_with_one_unreadable_file(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> object:
        if path == unreadable:
            raise PermissionError("fixture unreadable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_one_unreadable_file)

    sources = RepositoryScanner(Settings(_env_file=None)).scan(repository)

    assert [source.relative_path.as_posix() for source in sources] == ["readable.py"]


def test_invalid_gitignore_fails_clearly_instead_of_looking_like_empty_results(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    repository.mkdir()
    (repository / ".gitignore").write_bytes(b"\xff")
    (repository / "source.py").write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(RepositoryConfigurationError, match="Unable to read"):
        RepositoryScanner(Settings(_env_file=None)).scan(repository)


def test_symlinked_files_and_directories_are_not_followed(tmp_path: Path) -> None:
    repository = tmp_path / "project"
    external = tmp_path / "external"
    repository.mkdir()
    external.mkdir()
    (repository / "inside.py").write_text("x = 1\n", encoding="utf-8")
    (external / "outside.py").write_text("x = 2\n", encoding="utf-8")
    try:
        (repository / "linked.py").symlink_to(external / "outside.py")
        (repository / "linked_dir").symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")

    sources = RepositoryScanner(Settings(_env_file=None)).scan(repository)

    assert [source.relative_path.as_posix() for source in sources] == ["inside.py"]
    assert all(source.absolute_path.is_relative_to(repository.resolve()) for source in sources)
