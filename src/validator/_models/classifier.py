

from abc import ABC, abstractmethod
from typing import Any


class Classifier (ABC):
    max_length: int = 128

    @abstractmethod
    def evaluate(self, sequence: str|dict[str, Any]) -> bool:
        pass

    def get_max_length(self) -> int:
        return self.max_length