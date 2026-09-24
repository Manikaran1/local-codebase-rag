from __future__ import annotations

import math
from pathlib import Path

from codebase_rag.config.settings import Settings
from codebase_rag.domain.models import EmbeddingSpec
from codebase_rag.embeddings.base import Embedder
from codebase_rag.index.service import IndexingService
from codebase_rag.ingestion.repository_scanner import RepositoryScanner
from codebase_rag.processing import SourceProcessor
from codebase_rag.retrieval.hybrid import HybridRetriever

_SAMPLE_REPO = Path(__file__).parents[2] / "examples" / "sample_repo"


class ControlledSemanticEmbedder(Embedder):
    """Test-only semantic mapping for the sample fixture's four concepts."""

    def __init__(self) -> None:
        self.query_calls: list[str] = []

    @property
    def spec(self) -> EmbeddingSpec:
        return EmbeddingSpec(
            provider="controlled-test",
            model="sample-concepts-4d",
            revision="test-v1",
            normalized=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.casefold()
        concepts = [
            float(any(term in normalized for term in ("auth", "authenticate", "credential"))),
            float(
                any(
                    term in normalized
                    for term in ("database", "connection", "connect", "sqlite", " db ")
                )
            ),
            float(any(term in normalized for term in ("retry", "retries", "max_retries"))),
            float(
                any(
                    term in normalized
                    for term in ("documentation", "guide", "fixture repository")
                )
            ),
        ]
        if not any(concepts):
            concepts[3] = 1.0
        norm = math.sqrt(sum(value * value for value in concepts))
        return [value / norm for value in concepts]


def test_sample_repository_hybrid_retrieval_regressions(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", max_chunk_chars=400)
    embedder = ControlledSemanticEmbedder()
    scanner = RepositoryScanner(settings)
    processor = SourceProcessor(settings)
    indexing = IndexingService.from_data_dir(tmp_path / "data", embedder=embedder)

    sources = scanner.scan(_SAMPLE_REPO)
    for source in sources:
        text = source.absolute_path.read_text(encoding="utf-8")
        chunks = processor.process(source, text).chunks
        indexing.index_file(source.project_id, source.id, list(chunks))

    retriever = HybridRetriever(
        embedder=embedder,
        vector_index=indexing.vector_index,
        lexical_index=indexing.lexical_index,
    )
    project_id = sources[0].project_id

    single_source_cases = (
        ("Where is authentication handled?", "auth.py"),
        ("Where is the database connection created?", "database.py"),
        ("What controls retry count?", "config.py"),
        ("What is MAX_RETRIES set to?", "config.py"),
        ("Where is the fixture repository guide?", "README.md"),
    )
    for query, expected_file in single_source_cases:
        result = retriever.retrieve(project_id, query, limit=5)
        assert any(hit.chunk.relative_path.name == expected_file for hit in result.hits[:5]), query

    cross_file = retriever.retrieve(project_id, "Explain auth and DB interaction", limit=10)
    top_ten_files = {hit.chunk.relative_path.name for hit in cross_file.hits[:10]}
    assert {"auth.py", "database.py"} <= top_ten_files
    assert len(embedder.query_calls) == len(single_source_cases) + 1
