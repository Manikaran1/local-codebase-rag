"""Generation provider contract."""

from abc import ABC, abstractmethod


class Generator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
