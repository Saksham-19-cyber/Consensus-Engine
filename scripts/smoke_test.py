"""
Smoke test to measure LLM calls and latency per trial across all benchmark stages.
"""
import time
import sys
from dotenv import load_dotenv
load_dotenv()

from src.llm.client import get_total_calls, reset_total_calls
from src.eval.runner import run_engine_trial, evaluate_outcome
from src.eval.log_writer import TrialLogWriter
from scenarios.business_deal import BusinessDealScenario
from scenarios.strategic_negotiation import StrategicNegotiationScenario
from scenarios.from_text import run_custom_scenario

def smoke_test():
    print("=" * 60)
    print("CONSENSUS ENGINE: BENCHMARK SMOKE TEST (1 trial per stage)")
    print("=" * 60)

    # 1. Stage a: business_deal (single_text, consensus_engine with privacy probe)
    print("\n[Stage a] business_deal single_text + privacy probe (seed 42)...")
    sc = BusinessDealScenario()
    profiles, issues = sc.generate(seed=42)
    reset_total_calls()
    t0 = time.time()
    res_a = run_engine_trial(
        trial_id=42,
        scenario_name="business_deal",
        profiles=profiles,
        issues=issues,
        seed=42,
        protocol="single_text",
        model_config="120b_vs_20b",
        run_privacy_probe=True,
    )
    t_a = time.time() - t0
    calls_a = get_total_calls()
    print(f"  -> Done in {t_a:.1f}s, {calls_a} LLM calls. Agreement: {res_a.agreement_reached}, Rounds: {res_a.rounds_taken}")
    if res_a.privacy:
        print(f"  -> Privacy public lift: {res_a.privacy.public_transcript.lift_over_uniform if res_a.privacy.public_transcript else 'n/a'}")

    # 2. Stage b: business_deal (alternating_offers)
    print("\n[Stage b] business_deal alternating_offers (seed 42)...")
    reset_total_calls()
    t0 = time.time()
    res_b = run_engine_trial(
        trial_id=42,
        scenario_name="business_deal",
        profiles=profiles,
        issues=issues,
        seed=42,
        protocol="alternating_offers",
        model_config="120b_vs_20b",
        run_privacy_probe=False,
    )
    t_b = time.time() - t0
    calls_b = get_total_calls()
    print(f"  -> Done in {t_b:.1f}s, {calls_b} LLM calls. Agreement: {res_b.agreement_reached}, Rounds: {res_b.rounds_taken}")

    # 3. Stage c: strategic_negotiation (mixed, honest, strategic)
    print("\n[Stage c] strategic_negotiation mixed (seed 42)...")
    strat_sc = StrategicNegotiationScenario()
    strat_profiles, strat_issues = strat_sc.generate(seed=42)
    reset_total_calls()
    t0 = time.time()
    res_c = run_engine_trial(
        trial_id=42,
        scenario_name="strategic_negotiation",
        profiles=strat_profiles,
        issues=strat_issues,
        seed=42,
        protocol="single_text",
        model_config="120b_vs_20b",
        run_privacy_probe=True,
    )
    t_c = time.time() - t0
    calls_c = get_total_calls()
    print(f"  -> Done in {t_c:.1f}s, {calls_c} LLM calls. Agreement: {res_c.agreement_reached}, Rounds: {res_c.rounds_taken}")

    # 4. Stage f: Custom scenario check (software licensing)
    print("\n[Stage f] custom scenario CLI check...")
    text = (
        "FounderCo is licensing software to EnterpriseCorp. "
        "Issues: license fee ($50,000 to $500,000), contract duration (1 to 5 years), "
        "and support tier (1=basic, 2=standard, 3=dedicated). "
        "FounderCo needs cash flow so wants higher fees and 3-5 year contract. "
        "EnterpriseCorp has a strict budget under $200,000 but requires dedicated support (3)."
    )
    reset_total_calls()
    t0 = time.time()
    res_f = run_custom_scenario(description=text, seed=42, max_rounds=3, protocol="single_text")
    t_f = time.time() - t0
    calls_f = get_total_calls()
    print(f"  -> Done in {t_f:.1f}s, {calls_f} LLM calls. Feasible: {res_f['feasible']}, Agreed: {res_f['outcome'].agreement_reached}")

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY & FULL SUITE ESTIMATION")
    print("=" * 60)
    print(f"Stage a (single_text + privacy): {calls_a} calls/trial, {t_a:.1f}s")
    print(f"Stage b (alternating_offers):   {calls_b} calls/trial, {t_b:.1f}s")
    print(f"Stage c (strategic):            {calls_c} calls/trial, {t_c:.1f}s")
    print(f"Custom scenario pipeline:       {calls_f} calls/check, {t_f:.1f}s")

    # Full suite call estimation
    # Stage a: N=30 engine trials
    # Stage b: N=30 engine trials
    # Stage c: 3 variants (mixed, honest, strategic) * N=30 = 90 trials
    # Stage d: Reservation sweep (5 tiers * N=30 = 150) + Anchor sweep (4 levels * N=30 = 120) = 270 trials
    # Stage f: 4 checks
    total_engine_trials_30 = 30 + 30 + 90 + 270
    avg_calls_per_trial = (calls_a + calls_b + calls_c) / 3.0
    est_total_calls_30 = (total_engine_trials_30 * avg_calls_per_trial) + (4 * calls_f)
    print(f"\nEstimated total calls at N=30 ({total_engine_trials_30} engine trials): ~{est_total_calls_30:.0f} calls")

    if est_total_calls_30 > 5000:
        recommended_n = 20
        total_engine_trials_20 = 20 + 20 + 60 + (5 * 20 + 4 * 20)
        est_total_calls_20 = (total_engine_trials_20 * avg_calls_per_trial) + (4 * calls_f)
        print(f"--> Exceeds 5,000 calls! Reducing N uniformly to {recommended_n}.")
        print(f"Estimated total calls at N={recommended_n} ({total_engine_trials_20} trials): ~{est_total_calls_20:.0f} calls")
    else:
        recommended_n = 30
        print(f"--> Within 5,000 call budget. N={recommended_n} can be used.")

    return {
        "calls_a": calls_a,
        "calls_b": calls_b,
        "calls_c": calls_c,
        "calls_f": calls_f,
        "recommended_n": recommended_n,
    }

if __name__ == "__main__":
    smoke_test()
