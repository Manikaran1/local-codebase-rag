from __future__ import annotations

import shutil
from pathlib import Path

from codebase_rag.config.settings import Settings
from codebase_rag.domain.models import Chunk
from codebase_rag.index.service import IndexingService
from codebase_rag.ingestion.repository_scanner import RepositoryScanner
from codebase_rag.processing import SourceProcessor
from tests.fakes import DeterministicFakeEmbedder


def test_scan_process_index_replace_and_reopen_are_deterministic(tmp_path: Path) -> None:
    repository = tmp_path / "sample_repo"
    fixture = Path(__file__).parents[2] / "examples" / "sample_repo"
    shutil.copytree(fixture, repository)
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", max_chunk_chars=160)
    scanner = RepositoryScanner(settings)
    processor = SourceProcessor(settings)
    embedder = DeterministicFakeEmbedder(dimension=12)
    service = IndexingService.from_settings(settings, embedder=embedder)

    sources = scanner.scan(repository)
    chunks_by_file: dict[str, list[Chunk]] = {}
    project_id = sources[0].project_id
    for source in sources:
        result = processor.process(source, source.absolute_path.read_text(encoding="utf-8"))
        chunks_by_file[source.id] = list(result.chunks)
        service.index_file(source.project_id, source.id, list(result.chunks))

    target = next(source for source in sources if source.relative_path.as_posix() == "structure.py")
    old_chunks = chunks_by_file[target.id]
    assert old_chunks
    initial_ids = {chunk.id for chunks in chunks_by_file.values() for chunk in chunks}
    assert {row["chunk_id"] for row in service.store.vector_rows(project_id)} == initial_ids
    assert {row["chunk_id"] for row in service.store.lexical_rows(project_id)} == initial_ids

    old_query = embedder.embed_query(old_chunks[0].text)
    vector_hits = service.vector_index.search(project_id, old_query, limit=100)
    lexical_hits = service.lexical_index.search(project_id, "Repository.load.clean", limit=100)
    assert any(hit.chunk.id == old_chunks[0].id for hit in vector_hits)
    assert lexical_hits
    assert all(hit.chunk.project_id == project_id for hit in [*vector_hits, *lexical_hits])

    target.absolute_path.write_text(
        "def replacement_endpoint():\n    return 'phase4replacement'\n",
        encoding="utf-8",
    )
    rescanned = next(
        source
        for source in scanner.scan(repository)
        if source.relative_path.as_posix() == "structure.py"
    )
    new_chunks = list(
        processor.process(
            rescanned,
            rescanned.absolute_path.read_text(encoding="utf-8"),
        ).chunks
    )
    service.index_file(project_id, target.id, new_chunks)

    expected_ids = (initial_ids - {chunk.id for chunk in old_chunks}) | {
        chunk.id for chunk in new_chunks
    }
    vector_ids = {row["chunk_id"] for row in service.store.vector_rows(project_id)}
    lexical_ids = {row["chunk_id"] for row in service.store.lexical_rows(project_id)}
    assert vector_ids == expected_ids
    assert lexical_ids == expected_ids

    reopened = IndexingService.from_settings(
        settings,
        embedder=DeterministicFakeEmbedder(dimension=12),
    )
    query = embedder.embed_query(new_chunks[0].text)
    reopened_vector_hits = reopened.vector_index.search(project_id, query, limit=100)
    reopened_lexical_hits = reopened.lexical_index.search(
        project_id, "phase4replacement", limit=100
    )
    assert [hit.chunk.id for hit in reopened_vector_hits] == [
        hit.chunk.id for hit in service.vector_index.search(project_id, query, limit=100)
    ]
    assert [hit.chunk.id for hit in reopened_lexical_hits] == [
        hit.chunk.id
        for hit in service.lexical_index.search(project_id, "phase4replacement", limit=100)
    ]
    assert reopened_lexical_hits
    assert reopened_lexical_hits[0].vector_score is None
    assert reopened_lexical_hits[0].lexical_score is not None
