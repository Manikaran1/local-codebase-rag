from pathlib import Path


def test_fixture_repository_contains_expected_sources() -> None:
    fixture = Path("examples/sample_repo")
    assert (fixture / "auth.py").exists()
    assert (fixture / "database.py").exists()
    assert (fixture / "config.py").exists()
