from pathlib import Path

from codebase_rag.ingestion.repository_scanner import RepositoryScanner


def test_fixture_repository_produces_expected_manifest() -> None:
    fixture = Path("examples/sample_repo")
    sources = RepositoryScanner().scan(fixture)

    assert [source.relative_path.as_posix() for source in sources] == [
        "README.md",
        "auth.py",
        "config.py",
        "database.py",
        "nested/handlers.py",
    ]
    assert all(source.project_id for source in sources)
    assert all(source.content_hash for source in sources)
