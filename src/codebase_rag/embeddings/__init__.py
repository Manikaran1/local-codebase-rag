"""Local embedding provider interfaces and implementations."""

from codebase_rag.embeddings.base import Embedder
from codebase_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder

__all__ = ["Embedder", "SentenceTransformerEmbedder"]
