"""Deterministic discovery of indexable repository files."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from stat import S_ISREG

from codebase_rag.config.settings import Settings
from codebase_rag.domain.models import SourceFile
from codebase_rag.ingestion.filtering import (
    IgnoreSource,
    compile_ignore_spec,
    ignore_decision,
    is_binary_sample,
    load_gitignore,
)
from codebase_rag.ingestion.hashing import content_hash_stream, file_id_for, project_id_for

_BINARY_SAMPLE_BYTES = 8192
_QueuedDirectory = tuple[Path, PurePosixPath, tuple[IgnoreSource, ...]]


class RepositoryScanner:
    """Discover supported text files without following symbolic links."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else Settings()
        self._languages_by_extension = {
            _normalize_extension(extension): language
            for extension, language in self._settings.supported_extensions.items()
        }
        self._default_ignore_spec = compile_ignore_spec(
            self._settings.default_ignore_patterns,
            source_description="scanner default exclusions",
        )

    def scan(self, repository: Path) -> list[SourceFile]:
        """Return supported, readable text files in stable path order."""
        root = self._canonical_repository_root(repository)
        project_id = project_id_for(root)

        root_rules = load_gitignore(root, PurePosixPath())
        initial_rules: tuple[IgnoreSource, ...] = (
            (root_rules,) if root_rules is not None else ()
        )
        pending: list[_QueuedDirectory] = [(root, PurePosixPath(), initial_rules)]
        sources: list[SourceFile] = []

        while pending:
            directory, relative_directory, repository_rules = pending.pop()
            child_directories, directory_sources = self._scan_directory(
                directory=directory,
                relative_directory=relative_directory,
                repository_rules=repository_rules,
                root=root,
                project_id=project_id,
            )
            pending.extend(reversed(child_directories))
            sources.extend(directory_sources)

        return sorted(sources, key=lambda source: source.relative_path.as_posix())

    def _scan_directory(
        self,
        *,
        directory: Path,
        relative_directory: PurePosixPath,
        repository_rules: tuple[IgnoreSource, ...],
        root: Path,
        project_id: str,
    ) -> tuple[list[_QueuedDirectory], list[SourceFile]]:
        try:
            with os.scandir(directory) as entries_iterator:
                entries = sorted(entries_iterator, key=lambda entry: entry.name)
        except FileNotFoundError:
            return [], []
        except OSError as error:
            message = f"Unable to scan repository directory {directory}: {error}"
            raise OSError(message) from error

        child_directories: list[_QueuedDirectory] = []
        sources: list[SourceFile] = []
        for entry in entries:
            child_directory, source_file = self._scan_entry(
                entry=entry,
                relative_directory=relative_directory,
                repository_rules=repository_rules,
                root=root,
                project_id=project_id,
            )
            if child_directory is not None:
                child_directories.append(child_directory)
            if source_file is not None:
                sources.append(source_file)

        return child_directories, sources

    def _scan_entry(
        self,
        *,
        entry: os.DirEntry[str],
        relative_directory: PurePosixPath,
        repository_rules: tuple[IgnoreSource, ...],
        root: Path,
        project_id: str,
    ) -> tuple[_QueuedDirectory | None, SourceFile | None]:
        try:
            if entry.is_symlink():
                return None, None
            is_directory = entry.is_dir(follow_symlinks=False)
            is_file = not is_directory and entry.is_file(follow_symlinks=False)
        except OSError:
            return None, None
        if not is_directory and not is_file:
            return None, None

        relative_path = relative_directory / entry.name
        if entry.name == ".gitignore" and is_file:
            return None, None

        decision = ignore_decision(
            relative_path,
            is_directory=is_directory,
            repository_rules=repository_rules,
            default_rules=self._default_ignore_spec,
        )
        if decision.ignored_by_defaults or decision.ignored_by_repository:
            return None, None

        path = Path(entry.path)
        if not _is_scannable_path(path, root):
            return None, None
        if is_directory:
            nested_rules = load_gitignore(path, relative_path)
            child_rules = repository_rules
            if nested_rules is not None:
                child_rules += (nested_rules,)
            return (path, relative_path, child_rules), None

        source_file = self._scan_file_entry(
            entry=entry,
            path=path,
            root=root,
            relative_path=relative_path,
            project_id=project_id,
        )
        return None, source_file

    def _scan_file_entry(
        self,
        *,
        entry: os.DirEntry[str],
        path: Path,
        root: Path,
        relative_path: PurePosixPath,
        project_id: str,
    ) -> SourceFile | None:
        language = self._languages_by_extension.get(path.suffix.lower())
        if language is None:
            return None

        try:
            file_stat = entry.stat(follow_symlinks=False)
        except OSError:
            return None
        if not S_ISREG(file_stat.st_mode):
            return None
        if file_stat.st_size > self._settings.max_file_size_bytes:
            return None

        return self._source_file(
            path=path,
            root=root,
            relative_path=relative_path,
            project_id=project_id,
            language=language,
        )

    def _source_file(
        self,
        *,
        path: Path,
        root: Path,
        relative_path: PurePosixPath,
        project_id: str,
        language: str,
    ) -> SourceFile | None:
        max_size = self._settings.max_file_size_bytes
        sample_size = min(_BINARY_SAMPLE_BYTES, max_size + 1)
        try:
            with path.open("rb") as source:
                sample = source.read(sample_size)
                if len(sample) > max_size or is_binary_sample(sample):
                    return None
                result = content_hash_stream(
                    source,
                    initial_bytes=sample,
                    max_bytes=max_size,
                )
        except OSError:
            return None

        if result is None:
            return None
        content_hash, size_bytes = result
        path_text = relative_path.as_posix()
        relative_source_path = Path(*relative_path.parts)
        return SourceFile(
            id=file_id_for(project_id, path_text),
            project_id=project_id,
            relative_path=relative_source_path,
            absolute_path=root / relative_source_path,
            language=language,
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    @staticmethod
    def _canonical_repository_root(repository: Path) -> Path:
        try:
            root = repository.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Repository path does not exist: {repository}") from error
        except OSError as error:
            raise OSError(f"Unable to resolve repository path {repository}: {error}") from error
        if not root.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {repository}")
        return root


def _normalize_extension(extension: str) -> str:
    normalized = extension.lower()
    return normalized if normalized.startswith(".") else f".{normalized}"


def _is_scannable_path(path: Path, root: Path) -> bool:
    """Reject entries that resolve through links or escape the canonical root."""
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    return resolved == path and resolved.is_relative_to(root)
