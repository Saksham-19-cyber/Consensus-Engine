"""
Empirical Evaluation Scripts for:
1. Reservation Value Failure Rate Sweep (N=30 engine trials, N=100 baseline trials)
2. Strategic Anchor Inflation Sweep (N=30 engine trials per level, N=100 baseline trials)

Uses real engine runs logged to JSONL, with resumability.
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional

from scenarios.business_deal import BusinessDealScenario
from src.eval.baselines import naive_average_baseline, public_midpoint_baseline
from src.eval.log_writer import TrialLogWriter
from src.eval.runner import evaluate_outcome
from src.models.evaluation import TrialResult
from src.protocol.graph import run_negotiation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sweep_runner")


def run_reservation_sweep(
    n_engine: int = 30,
    n_baselines: int = 100,
    model_config: str = "20b_vs_20b",
    log_dir: str = "data/logs",
    run_engine: bool = True,
) -> dict:
    sc = BusinessDealScenario()
    thresholds = [0.40, 0.55, 0.60, 0.65, 0.70]
    log_path = Path(log_dir) / "reservation_sweep.jsonl"
    writer = TrialLogWriter(scenario_name="reservation_sweep", filepath=log_path)
    completed_keys = writer.get_completed_keys()

    results = {}
    print(f"\n{'='*70}\nRESERVATION VALUE IMPASSE & BREACH SWEEP (Engine N={n_engine}, Baselines N={n_baselines})\n{'='*70}")

    for r in thresholds:
        scenario_key = f"business_deal_r_{r:.2f}"
        print(f"\n--- Tier: r = {r:.2f} ---")

        # 1. Baseline runs across n_baselines
        mid_agreements = 0
        mid_breaches = 0
        naive_breaches = 0
        min_utils_mid = []

        for s in range(n_baselines):
            profiles, issues = sc.generate(seed=42 + s)
            for p in profiles:
                p.reservation_value = r

            p_mid = public_midpoint_baseline(profiles, issues)
            u_min = min(p.evaluate_proposal(p_mid) for p in profiles)
            min_utils_mid.append(u_min)
            if u_min >= r:
                mid_agreements += 1
            else:
                mid_breaches += 1

            p_naive = naive_average_baseline(profiles, issues)
            u_min_naive = min(p.evaluate_proposal(p_naive) for p in profiles)
            if u_min_naive < r:
                naive_breaches += 1

        mid_agree_pct = (mid_agreements / n_baselines) * 100.0
        mid_breach_pct = (mid_breaches / n_baselines) * 100.0
        naive_breach_pct = (naive_breaches / n_baselines) * 100.0
        mean_min_u = float(np.mean(min_utils_mid))

        print(f"Midpoint Agreement: {mid_agree_pct:5.1f}% | Midpoint Breach: {mid_breach_pct:5.1f}% | Naive Breach: {naive_breach_pct:5.1f}% | Mean Min U: {mean_min_u:.4f}")

        # 2. Real Engine runs across n_engine
        engine_agreements = 0
        engine_breaches = 0
        engine_trials_run = 0

        if run_engine:
            with writer:
                for s in range(n_engine):
                    seed = 42 + s
                    cache_key = (scenario_key, "consensus_engine", seed)
                    if cache_key in completed_keys:
                        logger.info("Skipping cached reservation trial %s", cache_key)
                        continue

                    profiles, issues = sc.generate(seed=seed)
                    for p in profiles:
                        p.reservation_value = r

                    from src.llm.client import get_total_calls
                    calls_before = get_total_calls()
                    outcome = run_negotiation(
                        profiles=profiles,
                        issues=issues,
                        max_rounds=5,
                        scenario_name=None,
                        model_config=model_config,
                        protocol="single_text",
                    )
                    trial_calls = get_total_calls() - calls_before

                    pareto, fairness, per_agent = evaluate_outcome(outcome.final_proposal, profiles, issues)

                    # Check breach: deal reached but any agent's true utility < reservation
                    breached = False
                    if outcome.agreement_reached and outcome.final_proposal:
                        breached = any(p.evaluate_proposal(outcome.final_proposal) < p.reservation_value for p in profiles)

                    if outcome.agreement_reached:
                        engine_agreements += 1
                    if breached:
                        engine_breaches += 1

                    trial_res = TrialResult(
                        trial_id=seed,
                        scenario_name=scenario_key,
                        method="consensus_engine",
                        protocol="single_text",
                        agreement_reached=outcome.agreement_reached,
                        rounds_taken=outcome.rounds_taken,
                        final_proposal=outcome.final_proposal,
                        per_agent_utilities=per_agent,
                        pareto=pareto,
                        fairness=fairness,
                        seed=seed,
                        llm_calls=trial_calls,
                        model_names={"mediator": model_config, "stakeholder": model_config},
                    )
                    writer.write(trial_res)
                    engine_trials_run += 1

        results[r] = {
            "midpoint_agreement_pct": mid_agree_pct,
            "midpoint_breach_pct": mid_breach_pct,
            "naive_breach_pct": naive_breach_pct,
            "mean_min_utility": mean_min_u,
            "engine_agreement_count": engine_agreements,
            "engine_breach_count": engine_breaches,
        }

    return results


def run_anchor_sweep(
    n_engine: int = 30,
    n_baselines: int = 100,
    model_config: str = "20b_vs_20b",
    log_dir: str = "data/logs",
    run_engine: bool = True,
) -> dict:
    sc = BusinessDealScenario()
    inflation_pcts = [0.10, 0.25, 0.50, 0.75]
    price_range = 100.0 - 10.0
    log_path = Path(log_dir) / "anchor_sweep.jsonl"
    writer = TrialLogWriter(scenario_name="anchor_sweep", filepath=log_path)
    completed_keys = writer.get_completed_keys()

    results = {}
    print(f"\n{'='*70}\nSTRATEGIC ANCHOR INFLATION SWEEP (Engine N={n_engine}, Baselines N={n_baselines})\n{'='*70}")

    for infl in inflation_pcts:
        scenario_key = f"business_deal_anchor_{int(infl*100)}"
        delta = infl * price_range
        print(f"\n--- Inflation +{int(infl*100)}% (Shift: +${delta:.2f}) ---")

        # 1. Baselines across n_baselines
        naive_shifts = []
        midpoint_shifts = []

        for s in range(n_baselines):
            profiles, issues = sc.generate(seed=42 + s)
            base_naive = naive_average_baseline(profiles, issues)["unit_price"]
            base_mid = public_midpoint_baseline(profiles, issues)["unit_price"]

            manipulated = []
            for p in profiles:
                p_copy = p.model_copy(deep=True)
                if p_copy.role == "Supplier":
                    p_copy.utility_function.ideal_values["unit_price"] += delta
                manipulated.append(p_copy)

            shift_naive = naive_average_baseline(manipulated, issues)["unit_price"] - base_naive
            shift_mid = public_midpoint_baseline(manipulated, issues)["unit_price"] - base_mid
            naive_shifts.append(shift_naive)
            midpoint_shifts.append(shift_mid)

        mean_naive = float(np.mean(naive_shifts))
        std_naive = float(np.std(naive_shifts))
        pct_naive = (mean_naive / price_range) * 100.0
        mean_mid = float(np.mean(midpoint_shifts))

        print(f"Naive Price Shift: +${mean_naive:5.2f} +/- ${std_naive:4.2f} ({pct_naive:4.1f}% capture) | Midpoint Shift: +${mean_mid:5.2f} (0.0% capture)")

        # 2. Real Engine runs across n_engine
        engine_bluff_flags = 0
        engine_price_shifts = []

        if run_engine:
            with writer:
                for s in range(n_engine):
                    seed = 42 + s
                    cache_key = (scenario_key, "consensus_engine", seed)
                    if cache_key in completed_keys:
                        logger.info("Skipping cached anchor trial %s", cache_key)
                        continue

                    # Run baseline trial without manipulation for delta calculation
                    profiles_base, issues = sc.generate(seed=seed)
                    # Manipulated profile for Supplier
                    profiles_manip, _ = sc.generate(seed=seed)
                    for p in profiles_manip:
                        if p.role == "Supplier":
                            p.utility_function.ideal_values["unit_price"] += delta

                    from src.llm.client import get_total_calls
                    calls_before = get_total_calls()
                    outcome = run_negotiation(
                        profiles=profiles_manip,
                        issues=issues,
                        max_rounds=5,
                        scenario_name=None,
                        model_config=model_config,
                        protocol="single_text",
                    )
                    trial_calls = get_total_calls() - calls_before

                    pareto, fairness, per_agent = evaluate_outcome(outcome.final_proposal, profiles_manip, issues)

                    # Check if mediator flagged bluffing
                    bluff_flagged = any(
                        any("BLUFF_SUSPECTED" in pattern for pattern in (msg.metadata or {}).get("patterns", []))
                        for msg in outcome.messages
                        if msg.metadata
                    )
                    if bluff_flagged:
                        engine_bluff_flags += 1

                    trial_res = TrialResult(
                        trial_id=seed,
                        scenario_name=scenario_key,
                        method="consensus_engine",
                        protocol="single_text",
                        agreement_reached=outcome.agreement_reached,
                        rounds_taken=outcome.rounds_taken,
                        final_proposal=outcome.final_proposal,
                        per_agent_utilities=per_agent,
                        pareto=pareto,
                        fairness=fairness,
                        seed=seed,
                        llm_calls=trial_calls,
                        model_names={"mediator": model_config, "stakeholder": model_config},
                    )
                    writer.write(trial_res)

        results[infl] = {
            "mean_naive_shift": mean_naive,
            "std_naive_shift": std_naive,
            "pct_naive_capture": pct_naive,
            "mean_midpoint_shift": mean_mid,
            "engine_bluff_flag_count": engine_bluff_flags,
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run empirical reservation and anchor sweeps")
    parser.add_argument("--n_engine", type=int, default=30, help="Number of engine trials per tier")
    parser.add_argument("--n_baselines", type=int, default=100, help="Number of baseline trials per tier")
    parser.add_argument("--model_config", type=str, default="fast", help="Model configuration name")
    parser.add_argument("--log_dir", type=str, default="data/logs", help="Directory for trial JSONL logs")
    parser.add_argument("--skip_engine", action="store_true", help="Run only no-LLM baseline sweeps")

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    run_reservation_sweep(
        n_engine=args.n_engine,
        n_baselines=args.n_baselines,
        model_config=args.model_config,
        log_dir=args.log_dir,
        run_engine=not args.skip_engine,
    )
    run_anchor_sweep(
        n_engine=args.n_engine,
        n_baselines=args.n_baselines,
        model_config=args.model_config,
        log_dir=args.log_dir,
        run_engine=not args.skip_engine,
    )
