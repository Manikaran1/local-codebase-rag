"""Authentication fixture."""


def authenticate(username: str, password: str) -> bool:
    """Return whether the supplied fixture credentials are accepted."""
    return username == "demo" and password == "local-only"
