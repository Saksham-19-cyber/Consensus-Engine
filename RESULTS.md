# Empirical Benchmark Results: Consensus Engine

> **Ground Rules**: Every table below was produced by real execution of the listed reproduction commands.
> No numbers were estimated, hand-written, or tuned.
> **Generated at**: 2026-11-01 19:41:04 UTC

## Table of Contents
- [Stage a: Fixed Benchmark (business_deal, single_text)](#stage-a-fixed-benchmark-business_deal-single_text)
- [Stage b: Protocol Comparison (single_text vs alternating_offers)](#stage-b-protocol-comparison-single_text-vs-alternating_offers)
- [Stage c: Strategic Misrepresentation Ablation (strategic_negotiation)](#stage-c-strategic-misrepresentation-ablation-strategic_negotiation)
- [Stage d: Reservation Impasse & Anchor Inflation Sweeps](#stage-d-reservation-impasse--anchor-inflation-sweeps)
- [Stage e: Privacy Leakage Reconstruction Probe](#stage-e-privacy-leakage-reconstruction-probe)
- [Stage f: Custom Scenario Checks (illustrative, N=3, NOT a benchmark)](#stage-f-custom-scenario-checks-illustrative-n3-not-a-benchmark)

---

## Stage a: Fixed Benchmark (business_deal, single_text)

Reproduction command:
```bash
python -m src.eval.runner --scenario business_deal --n_trials 30 --protocol single_text --model_config 20b_vs_20b --log_file results/benchmark_business_deal.jsonl
```

<!-- RESULTS:START stage_a_table -->
| Method | Agree% | Dist to Frontier (95% CI) | Nash Welfare (95% CI) | Utilitarian Welfare Ratio | Min Util | Gini | Rounds | Wilcoxon vs Engine | vs Engine |
|---|---|---|---|---|---|---|---|---|---|
| `consensus_engine` | 100.0% | 0.0654 [0.0544, 0.0780] | 0.3990 [0.3811, 0.4179] | 0.9156 [0.9022, 0.9274] | 0.6266 | 0.0753 | 1.0 | *(Engine)* | — |
| `naive_average` | 100.0% | 0.0362 [0.0288, 0.0441] | 0.4532 [0.4342, 0.4723] | 0.9525 [0.9425, 0.9612] | 0.6843 | 0.0651 | 0.0 | p = <1e-6 ** | better |
| `nash_bargaining` | 100.0% | 0.0350 [0.0310, 0.0391] | 0.5481 [0.5202, 0.5757] | 0.9991 [0.9977, 0.9999] | 0.7368 | 0.0504 | 0.0 | p = <1e-6 ** | better |
| `public_midpoint` | 100.0% | 0.0725 [0.0612, 0.0849] | 0.3885 [0.3744, 0.4019] | 0.9082 [0.8952, 0.9203] | 0.6193 | 0.0748 | 0.0 | p = 0.0017 ** | worse |
| `random_uniform` | 96.7% | 0.1610 [0.1281, 0.1973] | 0.2887 [0.2594, 0.3207] | 0.8205 [0.7912, 0.8480] | 0.5390 | 0.0826 | 0.0 | p = <1e-6 ** | worse |
<!-- RESULTS:END stage_a_table -->

## Stage b: Protocol Comparison (single_text vs alternating_offers)

Reproduction command:
```bash
python -m src.eval.runner --scenario business_deal --n_trials 30 --protocol alternating_offers --model_config 20b_vs_20b --log_file results/benchmark_alternating_offers.jsonl
```

*Not yet run (resume with `make benchmarks`)*

## Stage c: Strategic Misrepresentation Ablation (strategic_negotiation)

Reproduction command:
```bash
python -m src.eval.runner --scenario strategic_negotiation --n_trials 30 --protocol single_text --model_config 20b_vs_20b --log_file results/benchmark_strategic_negotiation.jsonl
```

*Not yet run (resume with `make benchmarks`)*

## Stage d: Reservation Impasse & Anchor Inflation Sweeps

Reproduction command:
```bash
python scripts/sweep_anchor_and_reservation.py --n_engine 30 --n_baselines 100 --model_config 20b_vs_20b
```

### Reservation Value Sweep (Failure Rate & Breach Rate)
> **Definition of Breach**: An outcome where an agreement was reached, but at least one agent's *true* utility fell below its reservation value.

<!-- RESULTS:START reservation_sweep_table -->
| Reservation ($r$) | Midpoint Failure% ($N=100$) | IR-gated Midpoint Agree% ($N=100$) | Naive Breach% ($N=100$) | Engine Agreement Rate | Engine Involuntary Breach Rate |
|---|---|---|---|---|---|
| $r = 0.40$ | 0.0% | 100.0% | 0.0% | 100.0% (2/2 trials) | 0.0% by construction (enforced by `would_accept`, covered by `tests/test_individual_rationality.py`) |
| $r = 0.55$ | 4.0% | 96.0% | 0.0% | *not yet run (resume with `python scripts/sweep_anchor_and_reservation.py`)* | 0.0% by construction (enforced by `would_accept`, covered by `tests/test_individual_rationality.py`) |
| $r = 0.60$ | 33.0% | 67.0% | 1.0% | *not yet run (resume with `python scripts/sweep_anchor_and_reservation.py`)* | 0.0% by construction (enforced by `would_accept`, covered by `tests/test_individual_rationality.py`) |
| $r = 0.65$ | 78.0% | 22.0% | 24.0% | *not yet run (resume with `python scripts/sweep_anchor_and_reservation.py`)* | 0.0% by construction (enforced by `would_accept`, covered by `tests/test_individual_rationality.py`) |
| $r = 0.70$ | 100.0% | 0.0% | 67.0% | *not yet run (resume with `python scripts/sweep_anchor_and_reservation.py`)* | 0.0% by construction (enforced by `would_accept`, covered by `tests/test_individual_rationality.py`) |
<!-- RESULTS:END reservation_sweep_table -->

### Strategic Anchor Inflation Sweep

<!-- RESULTS:START anchor_sweep_table -->
*Deterministic baseline arithmetic, no LLM*

| Inflation Level | Stated Price Shift ($\Delta P$) | Naive Baseline Price Shift | Naive Capture Rate | Midpoint Baseline Shift |
|---|---|---|---|---|
| **+10%** | +$9.00 | +$3.00 ± $0.00 | 3.3% | +$0.00 (0.0% capture) |
| **+25%** | +$22.50 | +$7.50 ± $0.00 | 8.3% | +$0.00 (0.0% capture) |
| **+50%** | +$45.00 | +$15.00 ± $0.00 | 16.7% | +$0.00 (0.0% capture) |
| **+75%** | +$67.50 | +$22.50 ± $0.00 | 25.0% | +$0.00 (0.0% capture) |
<!-- RESULTS:END anchor_sweep_table -->

## Stage e: Privacy Leakage Reconstruction Probe

Empirical reconstruction of private stakeholder utility weights from negotiation transcripts.
Reports cosine similarity of inferred weights to true weights, benchmarked against two reference baselines:
1. **Uniform Guess Baseline** (guess 1/K for each issue): Cosine ≈ 0.80 on 5-issue scenarios.
2. **Random Dirichlet Baseline** (random probability vector): Cosine ≈ 0.65–0.70.
Leakage is measured as lift over the uniform baseline: `(probe_cosine - uniform_cosine) / uniform_cosine`.

*Not yet run (requires `--run_privacy_probe`)*

## Stage f: Custom Scenario Checks (illustrative, N=3, NOT a benchmark)

Reproduction command:
```bash
python -m scenarios.from_text --text "<scenario description>" --rounds 3 --protocol single_text
```

### Check 1: Software Licensing (3 issues, 2 parties)
**Description**: *"FounderCo is licensing software to EnterpriseCorp. Issues: annual license fee ($50,000 to $500,000), contract duration (1 to 5 years), and support tier (1=basic, 2=standard, 3=dedicated). FounderCo strongly needs cash flow so they prefer higher fees and longer contracts with basic support. EnterpriseCorp has budget constraints preferring fees under $200,000, but insists on dedicated support (3)."*
- **Feasible**: YES
- **Agreement Reached**: True in 1 rounds
- **Package**: Annual License Fee: 260000.00 (Annual license fee in USD ($50,000 to $500,000).); Contract Duration Years: 3.00 (Contract duration in years (1 to 5).); Support Tier: 2.00 (Support tier: 1=basic, 2=standard, 3=dedicated.)
- **Distance to Frontier**: 0.1
- **Explanation**: The negotiation reached a voluntary agreement after 1 rounds. All parties achieved utility above their private reservation values, satisfying individual rationality. The final agreement achieves a Nash surplus product of 0.0850 and is within 0.1000 Euclidean distance from the Pareto frontier. This represents an efficient and mutually advantageous compromise.

### Check 2: Salary Negotiation (3 issues, 2 fully-specified parties)
**Description**: *"Alice and Bob are negotiating a compensation package for a senior engineer role. Issues: base salary ($120,000 to $200,000), equity shares (1,000 to 10,000 shares), and annual remote work days (0 to 100 days). Alice's priority is salary ($190,000 ideal, weight 60%), then equity (8,000 ideal, weight 30%), and remote days (50 ideal, weight 10%), reservation 0.60. Bob's company budget prefers base salary of $130,000 (weight 50%), equity of 3,000 shares (weight 30%), and remote days of 20 (weight 20%), reservation 0.55."*
- **Feasible**: YES
- **Agreement Reached**: True in 1 rounds
- **Package**: Base Salary: 165000.00 (Base salary in US dollars, ranging from $120,000 to $200,000.); Equity Shares: 6000.00 (Number of equity shares offered, ranging from 1,000 to 10,000 shares.); Annual Remote Work Days: 45.00 (Number of remote work days per year, ranging from 0 to 100 days.)
- **Distance to Frontier**: 0.0259
- **Explanation**: The negotiation reached a voluntary agreement after 1 rounds. All parties achieved utility above their private reservation values, satisfying individual rationality. The final agreement achieves a Nash surplus product of 0.0114 and is within 0.0259 Euclidean distance from the Pareto frontier. This represents an efficient and mutually advantageous compromise.

### Check 3: Urban Development (3 issues, 3 parties with 1 LLM-inferred)
**Description**: *"CityCouncil, DeveloperCorp, and NeighborhoodAssoc are negotiating a new downtown development project. Issues: building height (5 to 30 stories), park green space percentage (5% to 50%), and affordable housing units (10 to 100 units). CityCouncil prioritizes affordable housing (ideal 90 units, weight 50%) and green space (ideal 30%, weight 30%) with reservation 0.50. DeveloperCorp prioritizes height (ideal 28 stories, weight 60%) and lower affordable housing (ideal 15 units, weight 30%) with reservation 0.50. NeighborhoodAssoc represents local residents concerned about density and neighborhood character."*
- **Feasible**: YES
- **Agreement Reached**: True in 1 rounds
- **Package**: Building Height: 17.50 (Number of stories for the new downtown development (5 to 30).); Park Green Space Percentage: 27.50 (Percentage of the project area dedicated to park green space (5% to 50%).); Affordable Housing Units: 55.00 (Number of affordable housing units to be included in the project (10 to 100 units).)
- **Distance to Frontier**: 0.0212
- **Explanation**: The negotiation reached a voluntary agreement after 1 rounds. All parties achieved utility above their private reservation values, satisfying individual rationality. The final agreement achieves a Nash surplus product of 0.0083 and is within 0.0212 Euclidean distance from the Pareto frontier. This represents an efficient and mutually advantageous compromise.

### Check 4: Deliberately Infeasible Dispute (Reservation 0.95 each)
**Description**: *"PartyA and PartyB negotiate over transaction price ($10 to $100). PartyA refuses any deal giving less than 95% of the total surplus (reservation 0.95), and PartyB also strictly refuses any deal giving less than 95% of total surplus (reservation 0.95)."*
- **Feasible**: NO (Impasse detected)
- **Feasibility Warning**: ["Added second issue 'price_change_percentage' not explicitly mentioned; assumption made for trade-off.", 'Ideal values inferred; not specified by user.', 'Feasibility Warning: No deal can satisfy everyone as parsed (joint reservation surplus = -0.450 < 0). Consider lowering reservation thresholds or adjusting ideals before negotiating.']
