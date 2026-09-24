"""Construct the embedding provider selected by settings."""

from codebase_rag.config.settings import Settings
from codebase_rag.embeddings.base import Embedder
from codebase_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedder(
            settings.embedding_model,
            revision=settings.embedding_revision,
            allow_download=settings.embedding_allow_download,
            batch_size=settings.embedding_batch_size,
        )
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider!r}")
