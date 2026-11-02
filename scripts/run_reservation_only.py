import json
import logging
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

from scenarios.business_deal import BusinessDealScenario
from src.eval.runner import evaluate_outcome
from src.protocol.graph import run_negotiation
from src.models.evaluation import TrialResult
from src.llm.client import get_total_calls


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

    for rec in new_records_by_key.values():
        existing_lines.append(json.dumps(rec))

    with open(filepath, "w", encoding="utf-8") as f:
        for item in existing_lines:
            f.write(item + "\n")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def rerun_reservation():
    sc = BusinessDealScenario()
    seeds = [42, 43]
    r_val = 0.40
    scenario_key = f"business_deal_r_{r_val:.2f}"

    for idx, seed in enumerate(seeds, 1):
        profiles, issues = sc.generate(seed=seed)
        for p in profiles:
            p.reservation_value = r_val

        success = False
        while not success:
            try:
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
                for target_dir in [BASE_DIR / "results", BASE_DIR / "data" / "logs"]:
                    p_path = target_dir / "reservation_sweep.jsonl"
                    update_jsonl(p_path, {key: trial_res.model_dump()})
                success = True
            except Exception as e:
                print(f"Error for seed {seed}: {e}. Waiting 30s...", flush=True)
                time.sleep(30)

    print("Reservation sweep re-run successfully completed!", flush=True)

if __name__ == "__main__":
    rerun_reservation()

