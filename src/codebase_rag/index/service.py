"""Coordinate atomic file replacement across the local vector and lexical indexes."""

from __future__ import annotations

from pathlib import Path

from codebase_rag.config.settings import Settings
from codebase_rag.domain.models import Chunk
from codebase_rag.embeddings.base import Embedder
from codebase_rag.embeddings.factory import create_embedder
from codebase_rag.index.bm25 import TOKENIZER_VERSION, SQLiteBM25Index, tokenize
from codebase_rag.index.storage import SQLiteIndexStore
from codebase_rag.index.vector import SQLiteVectorIndex


class IndexingService:
    """Embed and replace one file's chunks in both indexes within one transaction."""

    def __init__(
        self,
        *,
        store: SQLiteIndexStore,
        embedder: Embedder,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.vector_index = SQLiteVectorIndex(store, embedder.spec)
        self.lexical_index = SQLiteBM25Index(store, k1=bm25_k1, b=bm25_b)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
    ) -> IndexingService:
        if settings.vector_backend != "sqlite":
            raise ValueError(f"Unsupported vector backend: {settings.vector_backend!r}")
        resolved_embedder = embedder if embedder is not None else create_embedder(settings)
        return cls(
            store=SQLiteIndexStore(settings.index_db_path),
            embedder=resolved_embedder,
            bm25_k1=settings.bm25_k1,
            bm25_b=settings.bm25_b,
        )

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path,
        *,
        embedder: Embedder,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> IndexingService:
        store = SQLiteIndexStore(data_dir / "indexes" / "index.sqlite3")
        return cls(store=store, embedder=embedder, bm25_k1=bm25_k1, bm25_b=bm25_b)

    def index_file(self, project_id: str, file_id: str, chunks: list[Chunk]) -> None:
        """Replace all chunks for a file, including clearing it when ``chunks`` is empty."""
        if not project_id.strip() or not file_id.strip():
            raise ValueError("project_id and file_id must not be empty")
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique in one file replacement")
        for chunk in chunks:
            self.store.validate_chunk(chunk)
            if chunk.project_id != project_id or chunk.file_id != file_id:
                raise ValueError("all chunks must belong to the requested project and file")

        if chunks:
            vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
            dimension, vector_blobs = self.vector_index._prepare_vectors(chunks, vectors)
            tokens = [tokenize(chunk.text) for chunk in chunks]
        else:
            dimension = None
            vector_blobs = []
            tokens = []

        self.store.replace_file(
            project_id=project_id,
            file_id=file_id,
            chunks=chunks,
            vectors=vector_blobs,
            dimension=dimension or None,
            tokens=tokens,
            embedding_spec=self.embedder.spec,
            tokenizer_version=TOKENIZER_VERSION,
            bm25_k1=self.lexical_index.k1,
            bm25_b=self.lexical_index.b,
        )

    def delete_file(self, project_id: str, file_id: str) -> int:
        return self.store.delete_file(project_id, file_id)

    def delete_project(self, project_id: str) -> int:
        return self.store.delete_project(project_id)
