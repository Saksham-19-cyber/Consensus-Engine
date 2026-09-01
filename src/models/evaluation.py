from __future__ import annotations
from pydantic import BaseModel, Field


class ParetoMetrics(BaseModel):
    distance_to_frontier: float = 0.0
    efficiency_ratio: float = 0.0
    is_pareto_optimal: bool = False
    social_welfare: float = 0.0


class FairnessMetrics(BaseModel):
    nash_welfare: float = 0.0
    min_utility: float = 0.0
    max_utility: float = 0.0
    gini_coefficient: float = 0.0
    envy_free: bool = False
    utility_spread: float = 0.0


class TrialResult(BaseModel):
    trial_id: int = 0
    scenario_name: str = ""
    method: str = ""
    agreement_reached: bool = False
    rounds_taken: int = 0
    per_agent_utilities: dict[str, float] = Field(default_factory=dict)
    final_proposal: dict[str, float] = Field(default_factory=dict)
    pareto: ParetoMetrics = Field(default_factory=ParetoMetrics)
    fairness: FairnessMetrics = Field(default_factory=FairnessMetrics)


class EvalReport(BaseModel):
    scenario_name: str = ""
    n_trials: int = 0
    methods: list[str] = Field(default_factory=list)
    results: list[TrialResult] = Field(default_factory=list)
    summary: dict[str, dict[str, float]] = Field(default_factory=dict)
