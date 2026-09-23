"""Deterministic embedding substitutes for tests and offline integration checks."""

from __future__ import annotations

import math
from hashlib import sha256

from codebase_rag.domain.models import EmbeddingSpec
from codebase_rag.embeddings.base import Embedder


class DeterministicFakeEmbedder(Embedder):
    """Generate normalized vectors from SHA-256 without process-randomized hashing."""

    def __init__(self, dimension: int = 8) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self.dimension = dimension

    @property
    def spec(self) -> EmbeddingSpec:
        return EmbeddingSpec(
            provider="fake-sha256", model=f"sha256-{self.dimension}-v1", normalized=True
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        raw: list[float] = []
        for index in range(self.dimension):
            digest = sha256(f"{index}\0{text}".encode()).digest()
            value = int.from_bytes(digest[:4], "big") / (2**32 - 1)
            raw.append(value * 2.0 - 1.0)
        norm = math.sqrt(sum(value * value for value in raw))
        return [value / norm for value in raw]
