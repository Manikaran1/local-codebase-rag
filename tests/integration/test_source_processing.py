from pathlib import Path

from codebase_rag.config.settings import Settings
from codebase_rag.ingestion.repository_scanner import RepositoryScanner
from codebase_rag.processing import SourceProcessor


def test_scanned_fixture_sources_flow_to_deterministic_chunks() -> None:
    fixture = Path("examples/sample_repo")
    sources = RepositoryScanner(Settings(_env_file=None)).scan(fixture)
    processor = SourceProcessor(Settings(_env_file=None, max_chunk_chars=160))

    python_results = []
    document_results = []
    for source in sources:
        text = source.absolute_path.read_text(encoding="utf-8")
        result = processor.process(source, text)
        repeated = processor.process(source, text)
        assert [chunk.id for chunk in result.chunks] == [chunk.id for chunk in repeated.chunks]
        for chunk in result.chunks:
            assert chunk.project_id == source.project_id
            assert chunk.file_id == source.id
            assert chunk.relative_path == source.relative_path
            assert 1 <= chunk.span.start_line <= chunk.span.end_line
            assert chunk.content_hash
            assert len(chunk.text) <= 160
        if source.relative_path.suffix == ".py":
            python_results.append(result)
        else:
            document_results.append(result)

    assert python_results
    assert document_results
    assert any(
        "Repository.load.clean" in chunk.metadata.get("contained_symbols", [])
        for result in python_results
        for chunk in result.chunks
    )
    assert any(
        chunk.metadata.get("heading_path")
        for result in document_results
        for chunk in result.chunks
    )
