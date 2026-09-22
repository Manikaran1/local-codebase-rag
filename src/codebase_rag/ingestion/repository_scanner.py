"""Repository discovery boundary."""

from pathlib import Path

from codebase_rag.domain.models import SourceFile


class RepositoryScanner:
    """Discover indexable files in a repository."""

    def scan(self, repository: Path) -> list[SourceFile]:
        raise NotImplementedError
