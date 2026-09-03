"""
Evaluation Runner
==================
Runs baseline and Consensus Engine trials, aggregates results with:
  - Bootstrap 95% confidence intervals per metric per method
  - Wilcoxon signed-rank tests between Consensus Engine and each baseline
  - Streaming JSONL log output for audit trail
  - Privacy leakage probe (optional, requires LLM call per trial)
  - Model size ablation via model_config parameter
"""
from __future__ import annotations
import logging
import time
import numpy as np
from collections import defaultdict
from typing import Callable, Any, Optional

from src.models.utility import StakeholderProfile
from src.models.evaluation import TrialResult, ParetoMetrics, FairnessMetrics, EvalReport, PrivacyMetrics
from src.eval.pareto import compute_pareto_frontier, pareto_distance, pareto_efficiency_ratio
from src.eval.fairness import compute_fairness_metrics
from src.eval.baselines import compute_baseline
from src.eval.log_writer import TrialLogWriter
from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration presets
# ---------------------------------------------------------------------------
MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "70b_vs_8b": {
        "negotiator_model": "llama-3.3-70b-versatile",
        "mediator_model": "llama-3.1-8b-instant",
    },
    "8b_vs_8b": {
        "negotiator_model": "llama-3.1-8b-instant",
        "mediator_model": "llama-3.1-8b-instant",
    },
    "70b_vs_70b": {
        "negotiator_model": "llama-3.3-70b-versatile",
        "mediator_model": "llama-3.3-70b-versatile",
    },
}


def _apply_model_config(config_name: str):
    """Temporarily override settings model fields (not thread-safe; batch runs only)."""
    config = MODEL_CONFIGS.get(config_name, {})
    if "negotiator_model" in config:
        settings.negotiator_model = config["negotiator_model"]
    if "mediator_model" in config:
        settings.mediator_model = config["mediator_model"]


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------
def _bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)]
    alpha = 1.0 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (round(lower, 4), round(upper, 4))


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------
def wilcoxon_test(
    values_a: list[float],
    values_b: list[float],
    alternative: str = "greater",
) -> dict[str, Any]:
    """
    Wilcoxon signed-rank test comparing two paired metric sequences.

    Returns dict with: statistic, p_value, significant_05, significant_01
    If scipy is not available, returns None fields gracefully.
    """
    try:
        from scipy.stats import wilcoxon
        if len(values_a) < 5 or len(values_b) < 5:
            return {"statistic": None, "p_value": None, "significant_05": None, "note": "too few trials"}
        n = min(len(values_a), len(values_b))
        stat, p = wilcoxon(values_a[:n], values_b[:n], alternative=alternative)
        return {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "significant_05": bool(p < 0.05),
            "significant_01": bool(p < 0.01),
        }
    except ImportError:
        logger.warning("scipy not installed — Wilcoxon test skipped")
        return {"statistic": None, "p_value": None, "significant_05": None, "note": "scipy_missing"}
    except Exception as e:
        logger.warning("wilcoxon_test failed: %s", e)
        return {"statistic": None, "p_value": None, "significant_05": None, "note": str(e)}


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_outcome(
    proposal: dict[str, float],
    profiles: list[StakeholderProfile],
    issues: list[dict],
    resolution: int = 40,
) -> tuple[ParetoMetrics, FairnessMetrics, dict[str, float]]:
    per_agent = {p.name: p.evaluate_proposal(proposal) for p in profiles}
    utility_functions = [p.utility_function for p in profiles]
    outcome_u = np.array(list(per_agent.values()))

    frontier = compute_pareto_frontier(utility_functions, issues, resolution=resolution)
    dist = pareto_distance(outcome_u, frontier) if len(frontier) > 0 else float("inf")
    ratio = pareto_efficiency_ratio(outcome_u, frontier) if len(frontier) > 0 else 0.0
    is_optimal = dist < 0.05

    pareto = ParetoMetrics(
        distance_to_frontier=round(dist, 4),
        efficiency_ratio=round(ratio, 4),
        is_pareto_optimal=is_optimal,
        social_welfare=round(float(np.sum(outcome_u)), 4),
    )

    fair_dict = compute_fairness_metrics(per_agent)
    fairness = FairnessMetrics(
        nash_welfare=round(fair_dict["nash_welfare"], 4),
        min_utility=round(fair_dict["min_utility"], 4),
        max_utility=round(fair_dict["max_utility"], 4),
        gini_coefficient=round(fair_dict["gini_coefficient"], 4),
        envy_free=fair_dict["envy_free"],
        utility_spread=round(fair_dict["utility_spread"], 4),
    )

    return pareto, fairness, per_agent


def run_baseline_trial(
    trial_id: int,
    method: str,
    profiles: list[StakeholderProfile],
    issues: list[dict],
    scenario_name: str,
    model_config_name: str = "70b_vs_8b",
) -> TrialResult:
    proposal = compute_baseline(method, profiles, issues)
    pareto, fairness, per_agent = evaluate_outcome(proposal, profiles, issues)
    all_accept = all(p.would_accept(proposal) for p in profiles)

    return TrialResult(
        trial_id=trial_id,
        scenario_name=scenario_name,
        method=method,
        protocol="baseline",
        agreement_reached=all_accept,
        rounds_taken=0,
        per_agent_utilities=per_agent,
        final_proposal=proposal,
        pareto=pareto,
        fairness=fairness,
        honesty_levels={p.name: p.honesty_level for p in profiles},
        model_config_name=model_config_name,
    )


def run_batch_baselines(
    scenario_generator: Callable[..., tuple[list[StakeholderProfile], list[dict]]],
    methods: list[str],
    n_trials: int,
    scenario_name: str,
    seed: int = 42,
    model_config_name: str = "70b_vs_8b",
    log_writer: Optional["TrialLogWriter"] = None,
) -> list[TrialResult]:
    _apply_model_config(model_config_name)
    results = []
    for trial_id in range(n_trials):
        profiles, issues = scenario_generator(seed=seed + trial_id)
        for method in methods:
            try:
                result = run_baseline_trial(
                    trial_id, method, profiles, issues, scenario_name, model_config_name
                )
                results.append(result)
                if log_writer:
                    log_writer.write(result)
                logger.info(
                    "trial=%d method=%s agreement=%s pareto_ratio=%.3f",
                    trial_id, method, result.agreement_reached, result.pareto.efficiency_ratio,
                )
            except Exception as e:
                logger.error("trial=%d method=%s error=%s", trial_id, method, e)
    return results


# ---------------------------------------------------------------------------
# Aggregation with CI and Wilcoxon
# ---------------------------------------------------------------------------
def aggregate_results(
    results: list[TrialResult],
    engine_method: str = "consensus_engine",
) -> dict[str, dict[str, Any]]:
    """
    Aggregate trial results per method with:
      - Mean ± 95% bootstrap CI for key metrics
      - Wilcoxon signed-rank test vs. the Consensus Engine (or specified engine_method)
    """
    by_method: dict[str, list[TrialResult]] = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    # Collect engine values for Wilcoxon comparisons
    engine_pareto = [t.pareto.efficiency_ratio for t in by_method.get(engine_method, [])]
    engine_nash = [t.fairness.nash_welfare for t in by_method.get(engine_method, [])]

    summary: dict[str, dict[str, Any]] = {}
    for method, trials in by_method.items():
        n = len(trials)
        agreed = [t for t in trials if t.agreement_reached]

        pareto_ratios = [t.pareto.efficiency_ratio for t in trials]
        nash_welfares = [t.fairness.nash_welfare for t in trials]
        min_utils = [t.fairness.min_utility for t in trials]
        ginis = [t.fairness.gini_coefficient for t in trials]
        rounds = [t.rounds_taken for t in trials if t.rounds_taken > 0]

        ci_pareto = _bootstrap_ci(pareto_ratios)
        ci_nash = _bootstrap_ci(nash_welfares)

        # Privacy metrics (only where available)
        privacy_scores = [
            t.privacy.mean_cosine_similarity
            for t in trials
            if t.privacy is not None
        ]

        # Bluff detection (only for mediated trials)
        bluff_flagged = sum(
            1 for t in trials
            if any(
                v.get("avg_satisfaction", 10) < 4.0 and v.get("avg_concession", 1.0) < 0.2
                for v in t.bluff_detection_scores.values()
            )
        )

        entry: dict[str, Any] = {
            "n_trials": n,
            "agreement_rate": round(len(agreed) / n, 4) if n > 0 else 0.0,
            "mean_pareto_ratio": round(float(np.mean(pareto_ratios)), 4) if pareto_ratios else 0.0,
            "std_pareto_ratio": round(float(np.std(pareto_ratios)), 4) if pareto_ratios else 0.0,
            "ci95_pareto_ratio": ci_pareto,
            "mean_nash_welfare": round(float(np.mean(nash_welfares)), 4) if nash_welfares else 0.0,
            "std_nash_welfare": round(float(np.std(nash_welfares)), 4) if nash_welfares else 0.0,
            "ci95_nash_welfare": ci_nash,
            "mean_min_utility": round(float(np.mean(min_utils)), 4) if min_utils else 0.0,
            "mean_gini": round(float(np.mean(ginis)), 4) if ginis else 0.0,
            "mean_rounds": round(float(np.mean(rounds)), 2) if rounds else 0.0,
            "mean_privacy_cosine": round(float(np.mean(privacy_scores)), 4) if privacy_scores else None,
            "bluff_detection_flagged": bluff_flagged,
        }

        # Wilcoxon vs Consensus Engine (only for non-engine methods)
        if method != engine_method and engine_pareto and pareto_ratios:
            entry["wilcoxon_pareto_vs_engine"] = wilcoxon_test(
                engine_pareto, pareto_ratios, alternative="greater"
            )
            entry["wilcoxon_nash_vs_engine"] = wilcoxon_test(
                engine_nash, nash_welfares, alternative="greater"
            )

        summary[method] = entry

    return summary
