"""File filtering helpers for repository discovery."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pathspec import GitIgnoreSpec


class RepositoryConfigurationError(ValueError):
    """Raised when repository ignore configuration cannot be read or parsed."""


@dataclass(frozen=True)
class IgnoreSource:
    """A compiled ignore file and the repository-relative directory it governs."""

    directory: PurePosixPath
    spec: GitIgnoreSpec
    has_negations: bool


@dataclass(frozen=True)
class IgnoreDecision:
    """Ignore outcome, keeping scanner defaults separate from repository rules."""

    ignored_by_repository: bool
    ignored_by_defaults: bool
    repository_may_reinclude_descendants: bool
    defaults_may_reinclude_descendants: bool


def compile_ignore_spec(
    patterns: Iterable[str],
    *,
    source_description: str,
) -> GitIgnoreSpec | None:
    """Compile Git ignore patterns, reporting malformed configuration clearly."""
    pattern_lines = tuple(patterns)
    if not pattern_lines:
        return None

    try:
        return GitIgnoreSpec.from_lines(pattern_lines)
    except ValueError as error:
        raise RepositoryConfigurationError(
            f"Invalid ignore patterns in {source_description}: {error}"
        ) from error


def load_gitignore(directory: Path, relative_directory: PurePosixPath) -> IgnoreSource | None:
    """Read one UTF-8 .gitignore file without following a symlink."""
    ignore_path = directory / ".gitignore"
    try:
        if ignore_path.is_symlink():
            return None
        contents = ignore_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as error:
        raise RepositoryConfigurationError(
            f"Unable to read repository ignore file {ignore_path}: {error}"
        ) from error

    spec = compile_ignore_spec(
        contents.splitlines(),
        source_description=str(ignore_path),
    )
    if spec is None:
        return None

    return IgnoreSource(
        directory=relative_directory,
        spec=spec,
        has_negations=any(pattern.include is False for pattern in spec.patterns),
    )


def ignore_decision(
    relative_path: PurePosixPath,
    *,
    is_directory: bool,
    repository_rules: tuple[IgnoreSource, ...],
    default_rules: GitIgnoreSpec | None,
    default_rules_have_negations: bool,
) -> IgnoreDecision:
    """Evaluate scoped .gitignore files and scanner defaults for one entry."""
    repository_ignored = False
    may_reinclude_descendants = False
    for source in repository_rules:
        scoped_path = relative_path.relative_to(source.directory).as_posix()
        if is_directory:
            scoped_path += "/"
        repository_ignored = _apply_spec(source.spec, scoped_path, repository_ignored)
        may_reinclude_descendants = may_reinclude_descendants or source.has_negations

    default_ignored = False
    if default_rules is not None:
        default_path = relative_path.as_posix()
        if is_directory:
            default_path += "/"
        default_ignored = _apply_spec(default_rules, default_path, False)

    return IgnoreDecision(
        ignored_by_repository=repository_ignored,
        ignored_by_defaults=default_ignored,
        repository_may_reinclude_descendants=may_reinclude_descendants,
        defaults_may_reinclude_descendants=default_rules_have_negations,
    )


def _apply_spec(spec: GitIgnoreSpec, path: str, initial_state: bool) -> bool:
    """Apply matching patterns in order while preserving an inherited state."""
    ignored = initial_state
    for pattern in spec.patterns:
        if pattern.match_file(path) is not None and pattern.include is not None:
            ignored = pattern.include
    return ignored


def is_binary_sample(sample: bytes) -> bool:
    """Conservatively identify binary data from a bounded byte sample."""
    if b"\0" in sample:
        return True
    if not sample:
        return False

    allowed_controls = {0x09, 0x0A, 0x0C, 0x0D}
    control_bytes = sum(byte < 0x20 and byte not in allowed_controls for byte in sample)
    return control_bytes / len(sample) > 0.30
