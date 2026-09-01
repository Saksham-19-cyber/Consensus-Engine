from __future__ import annotations
from abc import ABC, abstractmethod
from src.models.utility import StakeholderProfile, Issue


class Scenario(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def generate(self, seed: int = 42) -> tuple[list[StakeholderProfile], list[dict]]:
        ...

    def get_issues_meta(self, issues: list[Issue]) -> list[dict]:
        return [
            {"name": i.name, "range": [i.min_value, i.max_value], "description": i.description}
            for i in issues
        ]
