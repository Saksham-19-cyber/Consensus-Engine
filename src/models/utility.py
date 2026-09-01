from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
import numpy as np


class Issue(BaseModel):
    name: str
    min_value: float = 0.0
    max_value: float = 1.0
    description: str = ""

    def normalize(self, value: float) -> float:
        span = self.max_value - self.min_value
        if span == 0:
            return 0.5
        return (value - self.min_value) / span

    def denormalize(self, normalized: float) -> float:
        return self.min_value + normalized * (self.max_value - self.min_value)


class UtilityFunction(BaseModel):
    weights: dict[str, float] = Field(default_factory=dict)
    ideal_values: dict[str, float] = Field(default_factory=dict)
    issues: list[Issue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_weights(self):
        if self.weights:
            total = sum(self.weights.values())
            if abs(total - 1.0) > 0.01:
                factor = 1.0 / total
                self.weights = {k: v * factor for k, v in self.weights.items()}
        return self

    def score(self, proposal: dict[str, float]) -> float:
        if not self.weights:
            return 0.0
        issue_map = {i.name: i for i in self.issues}
        total = 0.0
        for issue_name, weight in self.weights.items():
            if issue_name not in proposal:
                continue
            issue = issue_map.get(issue_name)
            if not issue:
                continue
            ideal = self.ideal_values.get(issue_name, (issue.max_value + issue.min_value) / 2)
            normalized_val = issue.normalize(proposal[issue_name])
            normalized_ideal = issue.normalize(ideal)
            distance = abs(normalized_val - normalized_ideal)
            satisfaction = 1.0 - distance
            total += weight * satisfaction
        return round(max(0.0, min(1.0, total)), 4)

    def to_prompt_json(self) -> dict:
        return {
            "weights": self.weights,
            "ideal_values": self.ideal_values,
            "issues": [{"name": i.name, "range": [i.min_value, i.max_value]} for i in self.issues],
        }


class StakeholderProfile(BaseModel):
    name: str
    role: str = ""
    persona: str = ""
    utility_function: UtilityFunction
    reservation_value: float = Field(default=0.4, ge=0.0, le=1.0)

    def evaluate_proposal(self, proposal: dict[str, float]) -> float:
        return self.utility_function.score(proposal)

    def would_accept(self, proposal: dict[str, float]) -> bool:
        return self.evaluate_proposal(proposal) >= self.reservation_value
