"""Fixture demonstrating nested Python symbols."""
from __future__ import annotations


def normalize(value: str) -> str:
    return value.strip()


class Repository:
    def load(self, value: str) -> str:
        def clean(item: str) -> str:
            return item.strip()

        return clean(value)

    class Cache:
        def get(self, key: str) -> str:
            return key


@staticmethod
async def refresh(limit: int = 1) -> None:
    return None
