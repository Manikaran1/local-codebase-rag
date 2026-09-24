"""Errors raised by local embedding providers."""


class EmbeddingDependencyError(RuntimeError):
    """The configured embedding provider's optional dependency is unavailable."""


class EmbeddingModelError(RuntimeError):
    """The configured local embedding model could not be loaded or used."""
