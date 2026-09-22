"""Database fixture."""

import sqlite3


def connect(path: str = ":memory:") -> sqlite3.Connection:
    """Create the application's SQLite connection."""
    return sqlite3.connect(path)
