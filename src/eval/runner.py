"""
Evaluation Runner
==================
Runs baseline and Consensus Engine trials, aggregates results with:
  - Bootstrap 95% confidence intervals per metric per method
  - Seed-aligned Wilcoxon signed-rank tests between Consensus Engine and each baseline
  - Streaming JSONL log output for audit trail
  - Resumable runs: skips (scenario, method, seed) already in log; flushes after each trial
  - Privacy leakage probe (optional, requires LLM call per trial)
  - Model size ablation via model_config parameter
  - Full CLI via __main__
"""
from __future__ import annotations
import argparse
import logging
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any, Optional

import numpy as np

from src.models.utility import StakeholderProfile
from src.models.evaluation import TrialResult, ParetoMetrics, FairnessMetrics, EvalReport, PrivacyMetrics
from src.eval.pareto import (
    compute_pareto_frontier,
    pareto_distance,
    pareto_efficiency_ratio,
    utilitarian_welfare_ratio,
)
from src.eval.fairness import compute_fairness_metrics
from src.eval.baselines import compute_baseline
from src.eval.log_writer import TrialLogWriter, load_trial_log
from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration presets
# ---------------------------------------------------------------------------
MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "120b_vs_20b": {
        "negotiator_model": "openai/gpt-oss-120b",
        "mediator_model": "openai/gpt-oss-20b",
    },
    "20b_vs_20b": {
        "negotiator_model": "openai/gpt-oss-20b",
        "mediator_model": "openai/gpt-oss-20b",
    },
    "120b_vs_120b": {
        "negotiator_model": "openai/gpt-oss-120b",
        "mediator_model": "openai/gpt-oss-120b",
    },
    # Deprecated alias for 120b_vs_20b
    "70b_vs_8b": {
        "negotiator_model": "openai/gpt-oss-120b",
        "mediator_model": "openai/gpt-oss-20b",
    },
}


def _apply_model_config(config_name: str):
    """Temporarily override settings model fields (not thread-safe; batch runs only)."""
    config = MODEL_CONFIGS.get(config_name, {})
    if "negotiator_model" in config:
        settings.negotiator_model = config["negotiator_model"]
    if "mediator_model" in config:
        settings.mediator_model = config["mediator_model"]


def get_git_commit_hash() -> str:
    """Return the current short git commit hash, or 'unknown' if not in git."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


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
    alternative: str = "two-sided",
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
    resolution: int | None = None,
) -> tuple[ParetoMetrics, FairnessMetrics, dict[str, float]]:
    if resolution is None:
        n_iss = max(1, len(issues))
        resolution = max(5, min(20, int(10_000 ** (1.0 / n_iss))))

    per_agent = {p.name: p.evaluate_proposal(proposal) for p in profiles}
    utility_functions = [p.utility_function for p in profiles]
    outcome_u = np.array(list(per_agent.values()))

    frontier = compute_pareto_frontier(utility_functions, issues, resolution=resolution)
    dist = pareto_distance(outcome_u, frontier) if len(frontier) > 0 else float("inf")
    ratio = utilitarian_welfare_ratio(outcome_u, frontier) if len(frontier) > 0 else 0.0
    near_frontier = dist < 0.05

    pareto = ParetoMetrics(
        distance_to_frontier=round(dist, 4),
        efficiency_ratio=round(ratio, 4),
        utilitarian_welfare_ratio=round(ratio, 4),
        near_frontier=near_frontier,
        is_pareto_optimal=near_frontier,
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
    seed: int = 0,
    model_config_name: str = "120b_vs_20b",
    model_config: str | None = None,
    git_commit_hash: str | None = None,
) -> TrialResult:
    effective_config = model_config or model_config_name
    proposal = compute_baseline(method, profiles, issues)
    pareto, fairness, per_agent = evaluate_outcome(proposal, profiles, issues)
    all_accept = all(p.would_accept(proposal) for p in profiles)
    commit_hash = git_commit_hash or get_git_commit_hash()
    timestamp = datetime.now(timezone.utc).isoformat()
    models = MODEL_CONFIGS.get(
        effective_config,
        {"negotiator_model": settings.negotiator_model, "mediator_model": settings.mediator_model},
    )

    return TrialResult(
        trial_id=trial_id,
        scenario_name=scenario_name,
        seed=seed,
        method=method,
        protocol="baseline",
        agreement_reached=all_accept,
        rounds_taken=0,
        per_agent_utilities=per_agent,
        final_proposal=proposal,
        pareto=pareto,
        fairness=fairness,
        honesty_levels={p.name: p.honesty_level for p in profiles},
        model_config_name=effective_config,
        model_names=models,
        git_commit_hash=commit_hash,
        timestamp=timestamp,
    )


def run_engine_trial(
    trial_id: int,
    profiles: list[StakeholderProfile],
    issues: list[dict],
    scenario_name: str,
    seed: int = 0,
    protocol: str = "single_text",
    model_config_name: str = "120b_vs_20b",
    model_config: str | None = None,
    run_privacy_probe: bool = False,
    git_commit_hash: str | None = None,
) -> TrialResult:
    effective_config = model_config or model_config_name
    _apply_model_config(effective_config)
    from src.llm.client import get_total_calls, get_token_usage
    calls_before = get_total_calls()
    tokens_before = get_token_usage()

    from src.protocol.graph import run_negotiation
    # IMPORTANT: Do NOT pass scenario_name during benchmarks, so ChromaDB precedents don't contaminate independent trials
    outcome = run_negotiation(
        profiles=profiles,
        issues=issues,
        scenario_name=None,
        protocol=protocol,
        model_config=effective_config,
    )
    pareto, fairness, per_agent = evaluate_outcome(outcome.final_proposal, profiles, issues)

    privacy_metrics = None
    if run_privacy_probe and outcome.messages:
        try:
            from src.eval.privacy import run_privacy_probe as probe_fn
            privacy_metrics = probe_fn(
                outcome=outcome,
                profiles=profiles,
                issues=issues,
            )
        except Exception as e:
            logger.warning("privacy probe failed: %s", e)

    trial_calls = get_total_calls() - calls_before
    tokens_after = get_token_usage()
    trial_tokens = {
        "prompt_tokens": tokens_after["prompt_tokens"] - tokens_before["prompt_tokens"],
        "completion_tokens": tokens_after["completion_tokens"] - tokens_before["completion_tokens"],
        "total_tokens": tokens_after["total_tokens"] - tokens_before["total_tokens"],
    }
    commit_hash = git_commit_hash or get_git_commit_hash()
    timestamp = datetime.now(timezone.utc).isoformat()
    models = MODEL_CONFIGS.get(
        effective_config,
        {"negotiator_model": settings.negotiator_model, "mediator_model": settings.mediator_model},
    )

    return TrialResult(
        trial_id=trial_id,
        scenario_name=scenario_name,
        seed=seed,
        method="consensus_engine",
        protocol=protocol,
        agreement_reached=outcome.agreement_reached,
        rounds_taken=outcome.rounds_taken,
        per_agent_utilities=per_agent,
        final_proposal=outcome.final_proposal,
        pareto=pareto,
        fairness=fairness,
        privacy=privacy_metrics,
        honesty_levels={p.name: p.honesty_level for p in profiles},
        llm_calls=trial_calls,
        tokens_used=trial_tokens,
        model_config_name=effective_config,
        model_names=models,
        git_commit_hash=commit_hash,
        timestamp=timestamp,
        bluff_detection_scores=outcome.bluff_detection_scores,
    )


def run_batch_baselines(
    scenario_generator: Callable[..., tuple[list[StakeholderProfile], list[dict]]],
    methods: list[str],
    n_trials: int,
    scenario_name: str,
    seed: int = 42,
    model_config_name: str = "120b_vs_20b",
    log_writer: Optional["TrialLogWriter"] = None,
) -> list[TrialResult]:
    _apply_model_config(model_config_name)
    results = []
    completed_keys = log_writer.get_completed_keys() if log_writer else set()
    commit_hash = get_git_commit_hash()

    for trial_id in range(n_trials):
        current_seed = seed + trial_id
        profiles, issues = scenario_generator(seed=current_seed)
        for method in methods:
            if method == "consensus_engine":
                continue
            if (scenario_name, method, current_seed) in completed_keys:
                logger.info("Skipping already completed trial: scenario=%s method=%s seed=%d", scenario_name, method, current_seed)
                continue
            try:
                result = run_baseline_trial(
                    trial_id=trial_id,
                    method=method,
                    profiles=profiles,
                    issues=issues,
                    scenario_name=scenario_name,
                    seed=current_seed,
                    model_config_name=model_config_name,
                    git_commit_hash=commit_hash,
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


def run_batch_engine(
    scenario_generator: Callable[..., tuple[list[StakeholderProfile], list[dict]]],
    n_trials: int,
    scenario_name: str,
    seed: int = 42,
    protocol: str = "single_text",
    model_config_name: str = "120b_vs_20b",
    run_privacy_probe: bool = False,
    log_writer: Optional["TrialLogWriter"] = None,
) -> list[TrialResult]:
    _apply_model_config(model_config_name)
    results = []
    completed_keys = log_writer.get_completed_keys() if log_writer else set()
    commit_hash = get_git_commit_hash()

    for trial_id in range(n_trials):
        current_seed = seed + trial_id
        if (scenario_name, "consensus_engine", current_seed) in completed_keys:
            logger.info("Skipping already completed engine trial: scenario=%s seed=%d", scenario_name, current_seed)
            continue
        profiles, issues = scenario_generator(seed=current_seed)
        try:
            result = run_engine_trial(
                trial_id=trial_id,
                profiles=profiles,
                issues=issues,
                scenario_name=scenario_name,
                seed=current_seed,
                protocol=protocol,
                model_config_name=model_config_name,
                run_privacy_probe=run_privacy_probe,
                git_commit_hash=commit_hash,
            )
            results.append(result)
            if log_writer:
                log_writer.write(result)
            logger.info(
                "engine trial=%d agreement=%s rounds=%d pareto_ratio=%.3f",
                trial_id, result.agreement_reached, result.rounds_taken, result.pareto.efficiency_ratio,
            )
        except Exception as e:
            logger.error("engine trial=%d error=%s", trial_id, e)
    return results


# ---------------------------------------------------------------------------
# Aggregation with CI and Seed-Aligned Wilcoxon
# ---------------------------------------------------------------------------
def aggregate_results(
    results: list[TrialResult],
    engine_method: str = "consensus_engine",
) -> dict[str, dict[str, Any]]:
    """
    Aggregate trial results per method with:
      - Mean ± 95% bootstrap CI for key metrics
      - Seed-aligned Wilcoxon signed-rank test vs. the Consensus Engine (or specified engine_method)
    """
    by_method: dict[str, list[TrialResult]] = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    # Collect engine values mapped by seed (or trial_id) for aligned Wilcoxon tests
    engine_trials = by_method.get(engine_method, [])
    engine_by_seed: dict[int, TrialResult] = {}
    for t in engine_trials:
        key = t.seed if t.seed != 0 else t.trial_id
        engine_by_seed[key] = t

    summary: dict[str, dict[str, Any]] = {}
    for method, trials in by_method.items():
        n = len(trials)
        agreed = [t for t in trials if t.agreement_reached]

        pareto_ratios = [t.pareto.efficiency_ratio for t in trials]
        nash_welfares = [t.fairness.nash_welfare for t in trials]
        min_utils = [t.fairness.min_utility for t in trials]
        ginis = [t.fairness.gini_coefficient for t in trials]
        rounds = [t.rounds_taken for t in trials if t.rounds_taken > 0]
        distances = [t.pareto.distance_to_frontier for t in trials]

        ci_pareto = _bootstrap_ci(pareto_ratios)
        ci_nash = _bootstrap_ci(nash_welfares)
        ci_distance = _bootstrap_ci(distances)

        # Privacy metrics (only where available)
        privacy_scores = [
            t.privacy.mean_cosine_similarity
            for t in trials
            if t.privacy is not None and getattr(t.privacy, "mean_cosine_similarity", None) is not None
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
            "mean_utilitarian_welfare_ratio": round(float(np.mean(pareto_ratios)), 4) if pareto_ratios else 0.0,
            "ci95_utilitarian_welfare_ratio": ci_pareto,
            "mean_distance_to_frontier": round(float(np.mean(distances)), 4) if distances else 0.0,
            "ci95_distance_to_frontier": ci_distance,
            "mean_nash_welfare": round(float(np.mean(nash_welfares)), 4) if nash_welfares else 0.0,
            "std_nash_welfare": round(float(np.std(nash_welfares)), 4) if nash_welfares else 0.0,
            "ci95_nash_welfare": ci_nash,
            "mean_min_utility": round(float(np.mean(min_utils)), 4) if min_utils else 0.0,
            "mean_gini": round(float(np.mean(ginis)), 4) if ginis else 0.0,
            "mean_rounds": round(float(np.mean(rounds)), 2) if rounds else 0.0,
            "mean_privacy_cosine": round(float(np.mean(privacy_scores)), 4) if privacy_scores else None,
            "bluff_detection_flagged": bluff_flagged,
        }

        # Seed-aligned Wilcoxon test
        if method != engine_method and engine_by_seed:
            method_by_seed: dict[int, TrialResult] = {}
            for t in trials:
                key = t.seed if t.seed != 0 else t.trial_id
                method_by_seed[key] = t

            common_keys = sorted(set(engine_by_seed.keys()) & set(method_by_seed.keys()))
            if len(common_keys) >= 5:
                paired_engine_pareto = [engine_by_seed[k].pareto.efficiency_ratio for k in common_keys]
                paired_method_pareto = [method_by_seed[k].pareto.efficiency_ratio for k in common_keys]
                paired_engine_nash = [engine_by_seed[k].fairness.nash_welfare for k in common_keys]
                paired_method_nash = [method_by_seed[k].fairness.nash_welfare for k in common_keys]

                entry["wilcoxon_pareto_vs_engine"] = wilcoxon_test(
                    paired_engine_pareto, paired_method_pareto, alternative="two-sided"
                )
                entry["wilcoxon_nash_vs_engine"] = wilcoxon_test(
                    paired_engine_nash, paired_method_nash, alternative="two-sided"
                )
            else:
                entry["wilcoxon_pareto_vs_engine"] = {"statistic": None, "p_value": None, "significant_05": None, "note": "too few aligned trials"}
                entry["wilcoxon_nash_vs_engine"] = {"statistic": None, "p_value": None, "significant_05": None, "note": "too few aligned trials"}

        summary[method] = entry

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Consensus Engine Benchmark Runner")
    parser.add_argument("--scenario", type=str, default="business_deal", help="Scenario name")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials per method")
    parser.add_argument("--model_config", type=str, default="120b_vs_20b", help="Model configuration preset")
    parser.add_argument("--protocol", type=str, default="single_text", choices=["single_text", "alternating_offers"], help="Protocol for engine trials")
    parser.add_argument("--run_privacy_probe", action="store_true", help="Run post-hoc privacy leakage probe on engine transcripts")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["naive_average", "public_midpoint", "private_ideal_average", "nash_bargaining", "consensus_engine"],
        help="Methods to evaluate",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Directory to save JSONL trial logs")
    parser.add_argument("--log_file", type=str, default=None, help="Explicit JSONL file path to write/resume")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from scenarios.generator import make_scenario_generator
    generator = make_scenario_generator(args.scenario)

    log_dir = Path(args.log_dir) if args.log_dir else None
    log_file = Path(args.log_file) if args.log_file else None

    writer = TrialLogWriter(scenario_name=args.scenario, log_dir=log_dir, tag=args.protocol, filepath=log_file)
    writer.open()

    all_results: list[TrialResult] = []

    # If resuming from existing log file, preload existing trials for aggregation
    if writer.path.exists():
        from src.eval.log_writer import load_trial_log
        all_results.extend(load_trial_log(writer.path))

    baseline_methods = [m for m in args.methods if m != "consensus_engine"]
    if baseline_methods:
        logger.info("Running baseline methods: %s", baseline_methods)
        new_baselines = run_batch_baselines(
            scenario_generator=generator,
            methods=baseline_methods,
            n_trials=args.n_trials,
            scenario_name=args.scenario,
            seed=args.seed,
            model_config_name=args.model_config,
            log_writer=writer,
        )
        all_results.extend(new_baselines)

    if "consensus_engine" in args.methods:
        logger.info("Running consensus_engine trials (protocol=%s, model_config=%s)", args.protocol, args.model_config)
        new_engine = run_batch_engine(
            scenario_generator=generator,
            n_trials=args.n_trials,
            scenario_name=args.scenario,
            seed=args.seed,
            protocol=args.protocol,
            model_config_name=args.model_config,
            run_privacy_probe=args.run_privacy_probe,
            log_writer=writer,
        )
        all_results.extend(new_engine)

    writer.close()

    summary = aggregate_results(all_results)
    import pprint
    print("\n=== BENCHMARK SUMMARY ===")
    pprint.pprint(summary)


if __name__ == "__main__":
    main()
