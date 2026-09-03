from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


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


class PrivacyMetrics(BaseModel):
    """
    Empirical privacy leakage scores from the post-hoc reconstruction probe.

    mean_cosine_similarity: Average cosine similarity between inferred and true
        utility weights across all agents. Ranges [0, 1].
        - Near random baseline (~0.5–0.6 for 5 equally-weighted issues) = good privacy
        - Near 1.0 = transcript fully revealed agent preferences

    mean_kl_divergence: KL divergence D(true || inferred), averaged across agents.
        Higher values indicate better privacy (inferred is further from truth).

    per_agent_leakage: Per-agent {cosine_similarity, kl_divergence, leakage_score}
    """
    mean_cosine_similarity: float = 0.0
    mean_kl_divergence: float = 0.0
    per_agent_leakage: dict[str, dict[str, float]] = Field(default_factory=dict)
    n_agents: int = 0
    n_issues: int = 0


class TrialResult(BaseModel):
    trial_id: int = 0
    scenario_name: str = ""
    method: str = ""
    protocol: str = "single_text"
    agreement_reached: bool = False
    rounds_taken: int = 0
    per_agent_utilities: dict[str, float] = Field(default_factory=dict)
    final_proposal: dict[str, float] = Field(default_factory=dict)
    pareto: ParetoMetrics = Field(default_factory=ParetoMetrics)
    fairness: FairnessMetrics = Field(default_factory=FairnessMetrics)
    # Privacy leakage (None for baseline methods that don't produce transcripts)
    privacy: Optional[PrivacyMetrics] = None
    # Per-agent honesty levels (empty for honest baselines)
    honesty_levels: dict[str, float] = Field(default_factory=dict)
    # Model configuration used for this trial
    model_config_name: str = "70b_vs_8b"
    # Bluff detection scores (empty for non-mediated protocols)
    bluff_detection_scores: dict[str, dict[str, float]] = Field(default_factory=dict)


class EvalReport(BaseModel):
    scenario_name: str = ""
    n_trials: int = 0
    methods: list[str] = Field(default_factory=list)
    results: list[TrialResult] = Field(default_factory=list)
    summary: dict[str, dict] = Field(default_factory=dict)
