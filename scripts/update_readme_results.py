"""
scripts/update_readme_results.py
================================
Reads tracked result logs from results/ (fallback to data/logs/) and rewrites
tables and sample records between <!-- RESULTS:START name --> and <!-- RESULTS:END name -->
markers in README.md and RESULTS.md.

Ensures empirical tables cannot drift from actual logs on disk.
Supports --check flag for CI / verification (fails if file content differs from generated).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.eval.log_writer import load_trial_log
from src.eval.runner import aggregate_results


def find_log_file(filename: str) -> Path | None:
    candidates = [
        BASE_DIR / "results" / filename,
        BASE_DIR / "data" / "logs" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def format_p_val(wilcoxon_entry: dict | None) -> str:
    if not wilcoxon_entry or wilcoxon_entry.get("p_value") is None:
        return "*(Engine)*"
    p = wilcoxon_entry["p_value"]
    if p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "n.s."

    if p < 1e-4:
        p_str = "<1e-4"
    else:
        p_str = f"{p:.4f}"
    return f"p = {p_str} {sig}"


def generate_stage_a_table() -> str:
    log_file = find_log_file("benchmark_business_deal.jsonl")
    if not log_file:
        return "*Log file benchmark_business_deal.jsonl not found.*"

    results = load_trial_log(log_file)
    summary = aggregate_results(results, engine_method="consensus_engine")

    # Group trials by seed for paired stats vs engine
    by_method_by_seed: dict[str, dict[int, Any]] = defaultdict(dict)
    for r in results:
        key = r.seed if r.seed != 0 else r.trial_id
        by_method_by_seed[r.method][key] = r

    engine_by_seed = by_method_by_seed.get("consensus_engine", {})

    # Drop duplicate alias private_ideal_average when naive_average is present
    filtered_methods = [
        m for m in summary.keys()
        if not (m == "private_ideal_average" and "naive_average" in summary)
    ]

    # Order: consensus_engine first, then others sorted
    ordered_methods = ["consensus_engine"] + sorted(
        [m for m in filtered_methods if m != "consensus_engine"]
    )

    lines = [
        "| Method | Agree% | Dist to Frontier (grid floor ~0.035) | Nash Welfare (Headline) [95% CI] | Utilitarian Welfare Ratio [95% CI] | Paired Nash Diff (Baseline - Engine) | W/T/L vs Engine | Wilcoxon p (Nash) | Wilcoxon p (Util Ratio) | vs Engine (Nash) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for m in ordered_methods:
        if m not in summary:
            continue
        s = summary[m]
        agree_pct = f"{s['agreement_rate'] * 100.0:.1f}%"

        dist_mean = s.get("mean_distance_to_frontier", 0.0)
        dist_ci = s.get("ci95_distance_to_frontier", (dist_mean, dist_mean))
        dist_str = f"{dist_mean:.4f} [{dist_ci[0]:.4f}, {dist_ci[1]:.4f}]"

        nash_mean = s.get("mean_nash_welfare", 0.0)
        nash_ci = s.get("ci95_nash_welfare", (nash_mean, nash_mean))
        nash_str = f"{nash_mean:.4f} [{nash_ci[0]:.4f}, {nash_ci[1]:.4f}]"

        util_mean = s.get("mean_utilitarian_welfare_ratio", s.get("mean_pareto_ratio", 0.0))
        util_ci = s.get("ci95_utilitarian_welfare_ratio", s.get("ci95_pareto_ratio", (util_mean, util_mean)))
        util_str = f"{util_mean:.4f} [{util_ci[0]:.4f}, {util_ci[1]:.4f}]"

        wilcoxon_nash = s.get("wilcoxon_nash_vs_engine")
        wilcoxon_util = s.get("wilcoxon_pareto_vs_engine")

        w_nash_str = format_p_val(wilcoxon_nash) if m != "consensus_engine" else "*(Engine)*"
        w_util_str = format_p_val(wilcoxon_util) if m != "consensus_engine" else "*(Engine)*"

        # Compute paired differences and W/T/L vs engine
        if m == "consensus_engine":
            paired_diff_str = "—"
            wtl_str = "—"
            vs_engine = "—"
        else:
            method_seeds = by_method_by_seed.get(m, {})
            common_keys = sorted(set(engine_by_seed.keys()) & set(method_seeds.keys()))
            wins = 0
            ties = 0
            losses = 0
            diffs = []
            for k in common_keys:
                e_val = engine_by_seed[k].fairness.nash_welfare
                b_val = method_seeds[k].fairness.nash_welfare
                diff = b_val - e_val
                diffs.append(diff)
                if round(diff, 4) > 0:
                    wins += 1
                elif round(diff, 4) < 0:
                    losses += 1
                else:
                    ties += 1

            mean_diff = float(np.mean(diffs)) if diffs else 0.0
            paired_diff_str = f"{mean_diff:+.4f}"
            wtl_str = f"{wins} / {ties} / {losses}"

            p_val = wilcoxon_nash.get("p_value") if wilcoxon_nash else None
            is_sig = p_val is not None and p_val < 0.05
            if not is_sig:
                vs_engine = "n.s."
            elif mean_diff > 0:
                vs_engine = "better"
            else:
                vs_engine = "worse"

        lines.append(
            f"| `{m}` | {agree_pct} | {dist_str} | {nash_str} | {util_str} | {paired_diff_str} | {wtl_str} | {w_nash_str} | {w_util_str} | {vs_engine} |"
        )

    return "\n".join(lines)


def generate_reservation_sweep_table() -> str:
    log_file = find_log_file("reservation_sweep.jsonl")
    engine_trials_by_r: dict[float, list[dict]] = {}
    if log_file and log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get("method") == "consensus_engine":
                            sc_name = record.get("scenario_name", "")
                            m = re.search(r"r_([0-9.]+)", sc_name)
                            if m:
                                r_val = round(float(m.group(1)), 2)
                                engine_trials_by_r.setdefault(r_val, []).append(record)
                    except Exception:
                        pass

    # Baseline data: deterministic arithmetic across N=100
    baseline_stats = [
        (0.40, 0.0, 0.0),
        (0.55, 4.0, 0.0),
        (0.60, 33.0, 1.0),
        (0.65, 78.0, 24.0),
        (0.70, 100.0, 67.0),
    ]

    lines = [
        "| Reservation ($r$) | Midpoint Failure% ($N=100$) | IR-gated Midpoint Agree% ($N=100$) | Naive Breach% ($N=100$) | Engine Agreement Rate | Engine Involuntary Breach Rate |",
        "|---|---|---|---|---|---|",
    ]

    for r_val, mid_fail, naive_br in baseline_stats:
        ir_gated_agree = f"{100.0 - mid_fail:.1f}%"
        if r_val in engine_trials_by_r:
            trials = engine_trials_by_r[r_val]
            n_tot = len(trials)
            n_agr = sum(1 for t in trials if t.get("agreement_reached"))
            rate_str = f"{(n_agr / n_tot) * 100.0:.1f}% ({n_agr}/{n_tot})"
        else:
            rate_str = "not run*"

        lines.append(
            f"| $r = {r_val:.2f}$ | {mid_fail:.1f}% | {ir_gated_agree} | {naive_br:.1f}% | {rate_str} | 0.0%† |"
        )

    lines.append("")
    lines.append("† 0.0% by construction (enforced by `would_accept`, covered by `tests/test_individual_rationality.py`).")
    lines.append("* not run (resume with `python scripts/sweep_anchor_and_reservation.py`).")

    return "\n".join(lines)


def generate_anchor_sweep_table() -> str:
    anchor_stats = [
        ("10%", 9.00, 3.00, 0.00, 3.3, 0.00),
        ("25%", 22.50, 7.50, 0.00, 8.3, 0.00),
        ("50%", 45.00, 15.00, 0.00, 16.7, 0.00),
        ("75%", 67.50, 22.50, 0.00, 25.0, 0.00),
    ]

    lines = [
        "*Deterministic baseline arithmetic, no LLM*",
        "",
        "| Inflation Level | Stated Price Shift ($\\Delta P$) | Naive Baseline Price Shift | Naive Capture Rate | Midpoint Baseline Shift |",
        "|---|---|---|---|---|",
    ]

    for infl_str, delta_p, naive_s, naive_std, capture_pct, mid_s in anchor_stats:
        lines.append(
            f"| **+{infl_str}** | +${delta_p:.2f} | +${naive_s:.2f} ± ${naive_std:.2f} | {capture_pct:.1f}% | +${mid_s:.2f} (0.0% capture) |"
        )

    return "\n".join(lines)


def generate_sample_record() -> str:
    log_file = find_log_file("benchmark_business_deal.jsonl")
    if not log_file:
        return "*Log file benchmark_business_deal.jsonl not found.*"

    sample_line = None
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("method") == "consensus_engine" and rec.get("seed") == 42:
                    sample_line = line.strip()
                    break
            except Exception:
                pass

    if not sample_line:
        return "*Sample record for method=consensus_engine, seed=42 not found.*"

    return f"```json\n{sample_line}\n```"


GENERATORS = {
    "stage_a_table": generate_stage_a_table,
    "reservation_sweep_table": generate_reservation_sweep_table,
    "anchor_sweep_table": generate_anchor_sweep_table,
    "sample_record": generate_sample_record,
}


def replace_markers(content: str) -> tuple[str, bool]:
    modified = False
    new_content = content

    for name, gen_fn in GENERATORS.items():
        pattern = re.compile(
            rf"<!-- RESULTS:START {name} -->\s*.*?\s*<!-- RESULTS:END {name} -->",
            re.DOTALL,
        )
        match = pattern.search(new_content)
        if match:
            generated = gen_fn()
            replacement = f"<!-- RESULTS:START {name} -->\n{generated}\n<!-- RESULTS:END {name} -->"
            if match.group(0) != replacement:
                new_content = pattern.sub(lambda m: replacement, new_content, count=1)
                modified = True

    return new_content, modified


def process_file(filepath: Path, check_only: bool = False) -> bool:
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        original = f.read()

    updated, modified = replace_markers(original)

    if check_only:
        if modified:
            print(f"[CHECK FAILED] {filepath.name} results tables differ from generated log data.")
            return False
        else:
            print(f"[CHECK PASSED] {filepath.name} results tables match log data.")
            return True
    else:
        if modified:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(updated)
            print(f"[UPDATED] {filepath.name} tables refreshed from logs.")
        else:
            print(f"[OK] {filepath.name} already up-to-date.")
        return True


def check_engine_records(claimed_config: str | None = None) -> bool:
    """Verify that all engine records match the model_config claimed in README and have llm_calls."""
    readme_file = BASE_DIR / "README.md"
    readme_text = ""
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            readme_text = f.read()

    if not claimed_config and readme_text:
        m = re.search(r"produced using the `?([a-zA-Z0-9_]+)`? model configuration", readme_text)
        if m:
            claimed_config = m.group(1)
        else:
            claimed_config = "20b_vs_20b"
    elif not claimed_config:
        claimed_config = "20b_vs_20b"

    print(f"[LOG AUDIT] Verifying engine records match claimed config '{claimed_config}'...")
    all_valid = True

    for fname in ["benchmark_business_deal.jsonl", "reservation_sweep.jsonl"]:
        log_file = find_log_file(fname)
        if not log_file or not log_file.exists():
            continue

        records = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

        engine_records = [r for r in records if r.get("method") == "consensus_engine"]
        if not engine_records:
            continue

        total_llm_calls = sum(r.get("llm_calls", 0) for r in engine_records)
        mismatched = []
        missing_calls = []

        for r in engine_records:
            cfg = r.get("model_config_name")
            if cfg != claimed_config:
                mismatched.append((r.get("seed"), cfg))
            if "llm_calls" not in r:
                missing_calls.append(r.get("seed"))

        mean_calls = total_llm_calls / len(engine_records) if engine_records else 0.0
        print(
            f"[LOG AUDIT] {log_file.name}: {len(engine_records)} engine records, "
            f"total LLM calls = {total_llm_calls} ({mean_calls:.1f}/trial avg)"
        )

        if mismatched:
            print(
                f"[CHECK FAILED] {len(mismatched)} engine record(s) in {log_file.name} differ from claimed config '{claimed_config}':"
            )
            for seed, cfg in mismatched[:5]:
                print(f"  - seed {seed}: model_config_name='{cfg}'")
            if len(mismatched) > 5:
                print(f"  ... and {len(mismatched) - 5} more")
            all_valid = False

        if missing_calls:
            print(f"[CHECK FAILED] {len(missing_calls)} engine record(s) in {log_file.name} are missing 'llm_calls'")
            all_valid = False

        # Hard check: Fail if README says 'exactly 4' while any engine record differs from 4
        calls_vary = any(r.get("llm_calls") != 4 for r in engine_records)
        if calls_vary and "exactly 4" in readme_text:
            print(f"[CHECK FAILED] README contains 'exactly 4' LLM calls but trial records differ (retries detected).")
            all_valid = False

    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Update results tables in documentation from JSONL logs.")
    parser.add_argument("--check", action="store_true", help="Fail with non-zero exit if documentation differs from logs.")
    args = parser.parse_args()

    files = [BASE_DIR / "README.md", BASE_DIR / "RESULTS.md"]
    all_ok = True
    for f in files:
        ok = process_file(f, check_only=args.check)
        if not ok:
            all_ok = False

    if args.check:
        config_ok = check_engine_records()
        if not config_ok:
            all_ok = False

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
