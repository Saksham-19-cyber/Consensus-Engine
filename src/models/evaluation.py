from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class ParetoMetrics(BaseModel):
    distance_to_frontier: float = 0.0
    efficiency_ratio: float = 0.0  # Legacy alias for utilitarian_welfare_ratio
    utilitarian_welfare_ratio: float = 0.0
    near_frontier: bool = False  # True when distance_to_frontier < 0.05
    is_pareto_optimal: bool = False  # Legacy alias for backwards compatibility
    social_welfare: float = 0.0

    def __init__(self, **data):
        if "near_frontier" in data and "is_pareto_optimal" not in data:
            data["is_pareto_optimal"] = data["near_frontier"]
        elif "is_pareto_optimal" in data and "near_frontier" not in data:
            data["near_frontier"] = data["is_pareto_optimal"]
        super().__init__(**data)


class FairnessMetrics(BaseModel):
    nash_welfare: float = 0.0
    min_utility: float = 0.0
    max_utility: float = 0.0
    gini_coefficient: float = 0.0
    envy_free: bool = False
    utility_spread: float = 0.0


class PrivacyVariantMetrics(BaseModel):
    mean_cosine_similarity: Optional[float] = None
    lift_over_uniform: Optional[float] = None
    mean_kl_divergence: Optional[float] = None
    per_agent_leakage: dict[str, dict[str, float]] = Field(default_factory=dict)
    failure_count: int = 0
    probe_failed: bool = False


class PrivacyMetrics(BaseModel):
    """
    Empirical privacy leakage scores from the post-hoc reconstruction probe.
    Tracks two reference baselines (uniform-guess cosine and random-Dirichlet-guess cosine)
    and reports leakage as lift over uniform baseline.
    Tracks both public transcript and transcript with reasoning metadata.
    """
    mean_cosine_similarity: Optional[float] = None
    mean_kl_divergence: Optional[float] = None
    per_agent_leakage: dict[str, dict[str, float]] = Field(default_factory=dict)
    n_agents: int = 0
    n_issues: int = 0
    uniform_baseline_cosine: float = 0.0
    dirichlet_baseline_cosine: float = 0.0
    lift_over_uniform: Optional[float] = None
    failure_count: int = 0
    probe_failed: bool = False
    public_transcript: Optional[PrivacyVariantMetrics] = None
    with_reasoning: Optional[PrivacyVariantMetrics] = None


class TrialResult(BaseModel):
    trial_id: int = 0
    scenario_name: str = ""
    seed: int = 0
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
    model_config_name: str = "120b_vs_20b"
    model_names: dict[str, str] = Field(default_factory=dict)
    git_commit_hash: str = ""
    timestamp: str = ""
    llm_calls: int = 0
    tokens_used: dict[str, int] = Field(default_factory=dict)
    # Bluff detection scores (empty for non-mediated protocols)
    bluff_detection_scores: dict[str, dict[str, float]] = Field(default_factory=dict)


class EvalReport(BaseModel):
    scenario_name: str = ""
    n_trials: int = 0
    methods: list[str] = Field(default_factory=list)
    results: list[TrialResult] = Field(default_factory=list)
    summary: dict[str, dict] = Field(default_factory=dict)
