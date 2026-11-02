"""
scripts/rerun_affected_seeds.py
===============================
Re-runs the 18 affected seeds (42..59) for business_deal and 2 seeds (42, 43)
for reservation_sweep with --model_config 20b_vs_20b.
Replaces the old records and updates both results/ and data/logs/.
"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

from scenarios.business_deal import BusinessDealScenario
from src.eval.runner import run_engine_trial, evaluate_outcome
from src.protocol.graph import run_negotiation
from src.models.evaluation import TrialResult
from src.llm.client import get_total_calls

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rerun_seeds")


def update_jsonl(filepath: Path, new_records_by_key: dict[tuple, dict]):
    """Replace records in filepath matching (scenario_name, method, seed) with new ones."""
    if not filepath.exists():
        return
    existing_lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec.get("scenario_name"), rec.get("method"), rec.get("seed"))
            if key in new_records_by_key:
                existing_lines.append(json.dumps(new_records_by_key[key]))
                del new_records_by_key[key]
            else:
                existing_lines.append(line.strip())

    # Any remaining new records
    for rec in new_records_by_key.values():
        existing_lines.append(json.dumps(rec))

    with open(filepath, "w", encoding="utf-8") as f:
        for item in existing_lines:
            f.write(item + "\n")
    logger.info("Updated %s with replaced records.", filepath)


def rerun_business_deal_seeds():
    sc = BusinessDealScenario()
    seeds = list(range(42, 60))  # 42 to 59 inclusive (18 seeds)
    total_calls_all = 0
    new_results = {}

    print(f"--- Starting re-run of {len(seeds)} business_deal seeds with model_config=20b_vs_20b ---", flush=True)
    for idx, seed in enumerate(seeds, 1):
        profiles, issues = sc.generate(seed=seed)
        t0 = time.time()
        trial_res = run_engine_trial(
            trial_id=seed,
            profiles=profiles,
            issues=issues,
            scenario_name="business_deal",
            seed=seed,
            protocol="single_text",
            model_config="20b_vs_20b",
            run_privacy_probe=False,
        )
        elapsed = time.time() - t0
        total_calls_all += trial_res.llm_calls
        print(
            f"[{idx}/{len(seeds)}] seed={seed} done in {elapsed:.1f}s | "
            f"llm_calls={trial_res.llm_calls} | agreed={trial_res.agreement_reached} | "
            f"rounds={trial_res.rounds_taken} | config={trial_res.model_config_name}",
            flush=True,
        )
        key = ("business_deal", "consensus_engine", seed)
        new_results[key] = trial_res.model_dump()

        # Incremental save after each trial
        for target_dir in [BASE_DIR / "results", BASE_DIR / "data" / "logs"]:
            p = target_dir / "benchmark_business_deal.jsonl"
            update_jsonl(p, {key: trial_res.model_dump()})

    print(f"--- Finished business_deal seeds. Total LLM calls across 18 seeds: {total_calls_all} ---", flush=True)


def rerun_reservation_sweep_seeds():
    sc = BusinessDealScenario()
    seeds = [42, 43]
    r_val = 0.40
    scenario_key = f"business_deal_r_{r_val:.2f}"
    new_results = {}

    print(f"--- Starting re-run of reservation_sweep seeds ({seeds}) for r={r_val} with model_config=20b_vs_20b ---", flush=True)
    for idx, seed in enumerate(seeds, 1):
        profiles, issues = sc.generate(seed=seed)
        for p in profiles:
            p.reservation_value = r_val

        calls_before = get_total_calls()
        t0 = time.time()
        outcome = run_negotiation(
            profiles=profiles,
            issues=issues,
            max_rounds=5,
            scenario_name=None,
            model_config="20b_vs_20b",
            protocol="single_text",
        )
        trial_calls = get_total_calls() - calls_before
        pareto, fairness, per_agent = evaluate_outcome(outcome.final_proposal, profiles, issues)

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
            model_config_name="20b_vs_20b",
            model_names={"mediator": "20b_vs_20b", "stakeholder": "20b_vs_20b"},
        )
        elapsed = time.time() - t0
        print(
            f"[{idx}/{len(seeds)}] reservation seed={seed} done in {elapsed:.1f}s | "
            f"llm_calls={trial_res.llm_calls} | agreed={trial_res.agreement_reached} | "
            f"rounds={trial_res.rounds_taken} | config={trial_res.model_config_name}",
            flush=True,
        )
        key = (scenario_key, "consensus_engine", seed)
        new_results[key] = trial_res.model_dump()

        for target_dir in [BASE_DIR / "results", BASE_DIR / "data" / "logs"]:
            p = target_dir / "reservation_sweep.jsonl"
            update_jsonl(p, {key: trial_res.model_dump()})

    print("--- Finished reservation_sweep seeds. ---", flush=True)


if __name__ == "__main__":
    rerun_business_deal_seeds()
    rerun_reservation_sweep_seeds()

