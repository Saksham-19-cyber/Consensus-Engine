from __future__ import annotations
import asyncio
import logging
import time
import numpy as np
from typing import Callable, Any

from src.models.utility import StakeholderProfile
from src.models.evaluation import TrialResult, ParetoMetrics, FairnessMetrics, EvalReport
from src.eval.pareto import compute_pareto_frontier, pareto_distance, pareto_efficiency_ratio
from src.eval.fairness import compute_fairness_metrics
from src.eval.baselines import compute_baseline

logger = logging.getLogger(__name__)


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
) -> TrialResult:
    proposal = compute_baseline(method, profiles, issues)
    pareto, fairness, per_agent = evaluate_outcome(proposal, profiles, issues)

    all_accept = all(p.would_accept(proposal) for p in profiles)

    return TrialResult(
        trial_id=trial_id,
        scenario_name=scenario_name,
        method=method,
        agreement_reached=all_accept,
        rounds_taken=0,
        per_agent_utilities=per_agent,
        final_proposal=proposal,
        pareto=pareto,
        fairness=fairness,
    )


def run_batch_baselines(
    scenario_generator: Callable[..., tuple[list[StakeholderProfile], list[dict]]],
    methods: list[str],
    n_trials: int,
    scenario_name: str,
    seed: int = 42,
) -> list[TrialResult]:
    results = []
    for trial_id in range(n_trials):
        profiles, issues = scenario_generator(seed=seed + trial_id)
        for method in methods:
            try:
                result = run_baseline_trial(trial_id, method, profiles, issues, scenario_name)
                results.append(result)
                logger.info(
                    "trial=%d method=%s agreement=%s pareto_ratio=%.3f",
                    trial_id, method, result.agreement_reached, result.pareto.efficiency_ratio,
                )
            except Exception as e:
                logger.error("trial=%d method=%s error=%s", trial_id, method, e)
    return results


def aggregate_results(results: list[TrialResult]) -> dict[str, dict[str, float]]:
    from collections import defaultdict
    by_method = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    summary = {}
    for method, trials in by_method.items():
        n = len(trials)
        agreed = [t for t in trials if t.agreement_reached]

        pareto_ratios = [t.pareto.efficiency_ratio for t in trials]
        nash_welfares = [t.fairness.nash_welfare for t in trials]
        min_utils = [t.fairness.min_utility for t in trials]
        ginis = [t.fairness.gini_coefficient for t in trials]

        summary[method] = {
            "n_trials": n,
            "agreement_rate": len(agreed) / n if n > 0 else 0,
            "mean_pareto_ratio": float(np.mean(pareto_ratios)),
            "std_pareto_ratio": float(np.std(pareto_ratios)),
            "mean_nash_welfare": float(np.mean(nash_welfares)),
            "std_nash_welfare": float(np.std(nash_welfares)),
            "mean_min_utility": float(np.mean(min_utils)),
            "mean_gini": float(np.mean(ginis)),
        }

    return dict(summary)
