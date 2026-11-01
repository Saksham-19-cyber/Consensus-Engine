<div align="center">

<img src="assets/banner.jpg" alt="Consensus Engine — Multi-Agent Negotiation Under Private Information" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C8B6E?style=for-the-badge&logo=chainlink&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq%20SDK-F55036?style=for-the-badge&logo=openai&logoColor=white)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/Memory-ChromaDB-E85D04?style=for-the-badge&logo=databricks&logoColor=white)](https://trychroma.com)
[![Tests](https://img.shields.io/badge/Tests-132%20passing-2EA043?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-A371F7?style=for-the-badge)](LICENSE)

<br/>

**An autonomous multi-agent negotiation system built on LangGraph.**  
LLM-driven stakeholder agents with *private* utility functions and *structural privacy auditing* negotiate toward mutually acceptable agreements under strategic constraints — across two distinct protocols.

<br/>

[**Quickstart**](#-quickstart) · [**Architecture**](#-architecture) · [**Research**](#-research-contributions) · [**Protocols**](#-negotiation-protocols) · [**Evaluation**](#-evaluation-design) · [**API**](#-api-reference)

</div>

---

## ✨ What Makes This Different

Most "multi-agent negotiation" demos are cooperative solvers in disguise. Agents share preferences, the mediator splits the difference, everyone wins. That is not negotiation — it is arithmetic.

**Consensus Engine treats negotiation as it actually is:**

| Feature | Most frameworks | Consensus Engine |
|---|---|---|
| Agent preferences | Shared openly | Structurally private utility functions (audited for behavioral leakage) |
| Agent honesty | Always honest | Parameterised strategic misrepresentation *(implemented; not empirically evaluated)* |
| Protocol choice | One fixed flow | Mediated *or* Alternating-Offers |
| Result reporting | Point estimates | Mean ± 95% bootstrap CI + Wilcoxon tests |
| Audit trail | None | Full JSONL log per trial, re-aggregatable |
| Session memory | Stateless | ChromaDB precedents seeded into new sessions *(implemented; not empirically evaluated)* |

---

## 🧠 Research Contributions

### 1 · Strategic Misrepresentation *(Implemented; Not Empirically Evaluated)*
Each agent has a continuous `honesty_level ∈ [0, 1]`. Agents with `honesty_level < 1` are **strategic negotiators** — they anchor demands above their true ideal, suppress stated concession willingness in early rounds, and converge toward truth only as the deadline approaches. This is parameterised by the equation:

$$\hat{U}_i(\mathbf{x}, t) = U_i(\mathbf{x}) \cdot \left(1 - (1 - h_i) \cdot 0.3 \cdot \left(1 - \left(\frac{t}{T}\right)^2\right)\right)$$

The agent's **true** utility always governs the `acceptable` flag — strategic agents cannot bluff themselves into impasse. *(Note: Stage c strategic misrepresentation benchmark is not yet run).*

### 2 · Empirical Privacy Measurement & Behavioral Leakage
After each session, a **reconstruction probe LLM** reads the full dialogue transcript and attempts to infer each agent's utility weight distribution. Leakage is measured as:

$$\text{LeakScore}_i = \text{CosineSim}\left(\mathbf{w}_i,\ \hat{\mathbf{w}}_i\right)$$

Reference baselines are computed for comparison: a uniform-guess baseline (which averages ~0.80 for non-negative normalized weights on 5 issues) and a random Dirichlet baseline. Leakage is reported both as absolute cosine similarity and as lift over the uniform baseline. The framework empirically tests behavioral leakage across both public dialogue and transcripts containing private reasoning metadata. *(Note: Stage e probe run is not yet run).*

### 3 · Mediator Bluff Detection
The mediator maintains a rolling window of `(satisfaction_score, concession_willingness)` tuples per agent. Agents with persistently low satisfaction **and** low concession willingness are flagged as potential bluffers. Their names are injected into the revision prompt, causing the mediator to conserve concession budget until deadline pressure forces genuine signalling. *(Note: Requires $\ge 3$ rounds; has not triggered in 1-round benchmark runs).*

### 4 · Dual-Protocol Comparison
Two fundamentally different negotiation architectures run on identical scenarios, enabling controlled protocol comparison:

| Protocol | Description | Mediator? |
|---|---|---|
| `single_text` | Iterated mediated proposal-critique | ✅ Neutral mediator |
| `alternating_offers` | Direct turn-taking: Accept / Counteroffer / Walk-Away | ❌ None |

### 5 · Evaluation Method
All reported numbers include **95% bootstrap confidence intervals** (1,000 resamples) and **Wilcoxon signed-rank tests** vs. the Consensus Engine baseline across one scenario (`business_deal`) and one model config (`20b_vs_20b`). Every trial is streamed to auditable JSONL logs in `results/` and `data/logs/` — you can re-aggregate independently without re-running.

### 6 · Cross-Session Precedent Memory *(Implemented; Not Empirically Evaluated)*
A ChromaDB vector store persists negotiation outcomes after each session. On subsequent runs, the mediator's `propose_initial()` fetches semantically similar past proposals and uses them as soft anchors. The read/write loop is closed and unit-tested, though not yet evaluated in a multi-session ablation.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      run_negotiation()                          │
│                  src/protocol/graph.py                          │
│                                                                 │
│  protocol="single_text"          protocol="alternating_offers"  │
│         │                                     │                 │
│         ▼                                     ▼                 │
│  ┌─────────────────┐              ┌───────────────────────┐     │
│  │  LangGraph DAG  │              │  Alternating Offers   │     │
│  │  ┌───────────┐  │              │  Round Loop           │     │
│  │  │  Mediator │  │              │  ACCEPT / COUNTER /   │     │
│  │  │  Propose  │  │              │  WALK_AWAY            │     │
│  │  └─────┬─────┘  │              └───────────────────────┘     │
│  │        │        │                                            │
│  │  ┌─────▼─────┐  │      Stakeholder Agents                   │
│  │  │Stakeholder│  │  ┌────────────────────────────────────┐   │
│  │  │ Critique  │◄─┼──┤ honesty_level=1.0 → StakeholderAgent│  │
│  │  └─────┬─────┘  │  │ honesty_level<1.0 → StrategicAgent  │  │
│  │        │        │  └────────────────────────────────────┘   │
│  │  ┌─────▼─────┐  │                                           │
│  │  │   Check   │  │      Memory                               │
│  │  │Termination│  │  ┌────────────────────────────────────┐   │
│  │  └───────────┘  │  │ ChromaDB: read precedents at start  │  │
│  └─────────────────┘  │ ChromaDB: write outcome at end      │  │
│                       └────────────────────────────────────┘   │
│                                                                 │
│  Post-run: Privacy Probe → PrivacyMetrics (cosine + KL div.)   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Map

```
src/
├── agents/
│   ├── stakeholder.py             # Honest agent: generate_critique()
│   ├── strategic_stakeholder.py   # Bluffing agent: stated_utility deflation
│   └── mediator.py                # Proposal + bluff detection + precedents
│
├── protocol/
│   ├── graph.py                   # LangGraph DAG + agent auto-routing
│   ├── alternating_offers.py      # Direct offer-counteroffer protocol
│   ├── state.py                   # NegotiationState TypedDict
│   └── rules.py                   # Termination conditions
│
├── eval/
│   ├── runner.py                  # Batch evaluation: CI, Wilcoxon, model configs
│   ├── privacy.py                 # Reconstruction probe → leakage metrics
│   ├── log_writer.py              # Streaming JSONL trial logs
│   ├── report.py                  # mean±CI Markdown + JSON reports
│   ├── pareto.py                  # Pareto frontier + efficiency ratio
│   ├── fairness.py                # Nash welfare, Gini coefficient
│   └── baselines.py               # Naive average, Nash bargaining solution
│
├── models/
│   ├── utility.py                 # StakeholderProfile, UtilityFunction, Issue
│   ├── negotiation.py             # MediatorResponse, Critique, CounterOffer
│   └── evaluation.py              # TrialResult, PrivacyMetrics, EvalReport
│
├── llm/
│   ├── client.py                  # Groq structured_completion + plain_completion
│   └── prompts.py                 # All prompt builders (including strategic)
│
└── memory/
    └── store.py                   # ChromaDB read/write (retrieve + store outcomes)

scenarios/
├── roommate.py                    # 2-party: rent, cleaning, noise, guests
├── business_deal.py               # 3-party: price, volume, delivery, payment, quality
├── trip_planning.py               # 3–5 party: budget, destination, duration, hotel
└── strategic_negotiation.py       # 3-party with honesty_level ~ Uniform(0.3, 1.0)
```

---

## 🔬 Mathematical Formulation

### Private Utility Function

For agent $i$ evaluating a proposal vector $\mathbf{x} = (x_1, \ldots, x_K)$ over $K$ issues with ranges $[x_k^{\min}, x_k^{\max}]$:

$$U_i(\mathbf{x}) = \sum_{k=1}^{K} w_{i,k} \cdot \left(1 - \frac{|x_k - x_{i,k}^{\ast}|}{x_k^{\max} - x_k^{\min}}\right), \quad \sum_{k=1}^K w_{i,k} = 1$$

Weights $w_{i,k}$ and ideal values $x_{i,k}^{\ast}$ are **never transmitted** — only communicated indirectly through strategic language.

### Nash Social Welfare

As implemented in `src/eval/fairness.py` (`nash_social_welfare`), Nash social welfare is computed as the product of agent utilities:

$$\text{NSW}(\mathbf{x}^{\ast}) = \prod_{i=1}^{N} \max\left(U_i(\mathbf{x}^{\ast}),\ \varepsilon\right)$$

where $\varepsilon = 10^{-10}$.

### Utilitarian Welfare Ratio & Distance to Frontier

The **Utilitarian Welfare Ratio** measures the achieved sum of utilities relative to the maximum possible sum on the Pareto frontier:

$$\text{UWR}(\mathbf{x}^{\ast}) = \frac{\sum_{i=1}^{N} U_i(\mathbf{x}^{\ast})}{\displaystyle\max_{\mathbf{u} \in \mathcal{F}} \sum_{i=1}^{N} u_i}$$

The **Distance to Frontier** measures the Euclidean distance from the achieved utility vector $\mathbf{u}(\mathbf{x}^{\ast})$ to the nearest point on the Pareto frontier $\mathcal{F}$ in $[0, 1]^N$ utility space:

$$\text{DistToFrontier}(\mathbf{x}^{\ast}) = \min_{\mathbf{y} \in \mathcal{F}} \|\mathbf{u}(\mathbf{x}^{\ast}) - \mathbf{y}\|_2$$

where $\mathcal{F}$ is the Pareto frontier computed by exhaustive grid search ($\le 4$ issues) or Monte Carlo sampling (5–6 issues).

### Bluff Detection Heuristic

Agent $i$ is flagged as a bluff suspect after round $t$ if, over the last $W=3$ rounds:

$$\bar{s}_i < \tau_s \;\wedge\; \bar{c}_i < \tau_c$$

where $\bar{s}_i$ is mean satisfaction score ($\tau_s = 4.0$) and $\bar{c}_i$ is mean concession willingness ($\tau_c = 0.2$).

---

## 🤝 Negotiation Protocols

### Single-Text Mediation (`protocol="single_text"`)

The default LangGraph-orchestrated flow:

```
Round 1:  Mediator → Initial Proposal (seeded from ChromaDB precedents)
Round t:  Stakeholders → Critiques (strategic agents deflate stated utility)
          Mediator → Bluff Detection → Revised Proposal or Agreement/Impasse
Round T:  MAX_ROUNDS → outcome recorded regardless
```

The mediator uses **issue-linkage**: if agent A wants issue X higher and agent B wants issue Y higher, it proposes a trade. This is surfaced via the `issue_linkage` field of `MediatorResponse`.

### Alternating Offers (`protocol="alternating_offers"`)

Direct bilateral/multilateral offer exchange — no mediator:

```
Round 1:  Random agent → Initial Offer (midpoint)
Round t:  All other agents → ACCEPT / COUNTEROFFER / WALK_AWAY
          If counteroffers: average → new proposal, rotate proposer
          If all ACCEPT   → AGREED
          If any WALK_AWAY → IMPASSE
Round T:  MAX_ROUNDS → IMPASSE
```

The `acceptable` flag is always determined by **true utility** — strategic agents cannot accept below their reservation value regardless of bluffing posture.

---

## 🧪 Evaluation Design

### Running Benchmarks

```bash
# N=30 trials across all baselines + Consensus Engine on business_deal
python -m src.eval.runner \
  --scenario business_deal \
  --n_trials 30 \
  --protocol single_text \
  --model_config 20b_vs_20b \
  --log_file results/benchmark_business_deal.jsonl

# Re-aggregate from saved JSONL without re-running
python -c "
from src.eval.log_writer import load_trial_log
from src.eval.runner import aggregate_results
from src.eval.report import generate_markdown_report
from pathlib import Path

results = load_trial_log(Path('results/benchmark_business_deal.jsonl'))
summary = aggregate_results(results, engine_method='consensus_engine')
print(generate_markdown_report('business_deal', summary, results))
"
```

### Model Configurations

| Config | Stakeholder Model | Mediator Model | Use Case |
|---|---|---|---|
| `20b_vs_20b` | `openai/gpt-oss-20b` | `openai/gpt-oss-20b` | **Used for published benchmark runs** (N=30 Stage a) |
| `120b_vs_20b` | `openai/gpt-oss-120b` | `openai/gpt-oss-20b` | Asymmetric high intelligence |
| `120b_vs_120b` | `openai/gpt-oss-120b` | `openai/gpt-oss-120b` | Maximum capability benchmark |

> **Note**: All published empirical results in this repository were produced using the `20b_vs_20b` model configuration.

### Baselines

| Baseline | Description | Information Access |
|---|---|---|
| `public_midpoint` | Midpoint of each public issue range $[x_k^{\min}, x_k^{\max}]$ | Public domain bounds only |
| `naive_average` | Arithmetic mean of agents' private ideal points | Privileged semi-oracle (knows all private targets) |
| `random_uniform` | Random uniform coordinate sampled within issue bounds | Public domain bounds only |
| `nash_bargaining` | Maximise $\prod_i \max(U_i - r_i, 0)$ via exhaustive grid search | Full-information cooperative oracle |
| `single_llm_oracle` | Single LLM prompted with full private profiles to propose an outcome | Full-information unmediated oracle |

---

## 📊 Concrete Execution Outputs

The following outputs and benchmark tables are drawn from tracked execution runs across our evaluation suite.

### 1 · Multi-Agent Strategic Negotiation Session (Stage c)

*Not yet run in batch benchmarks (resume with `python -m src.eval.runner --scenario strategic_negotiation --n_trials 30 --protocol single_text --model_config 20b_vs_20b --log_file results/benchmark_strategic_negotiation.jsonl`).*

### 2 · Empirical Privacy Probe: The Gap Between Structural Privacy & Behavioral Leakage

A central architectural pillar of multi-agent negotiation frameworks is **structural privacy**: raw utility functions, numeric weights, and reservation thresholds are never passed across agents or transmitted in message payloads.

However, our post-hoc **reconstruction probe evaluates behavioral leakage** against empirical baselines: a uniform guess (which scores ~0.80 cosine similarity for non-negative normalized weights on 5 issues) and a random Dirichlet baseline. Leakage is measured as lift over the uniform baseline:

> ⚠️ **Evaluating Behavioral Disclosure:**  
> The probe assesses how much additional preference structure is disclosed beyond reference baselines. Because negotiation requires communicating trade-offs ("*I cannot accept 20-day delivery unless price drops*"), natural language can expose weight priorities even when raw parameters are never transmitted.

*Not yet run in published benchmarks (requires running with `--run_privacy_probe`).*

### 3 · Real-Time Mediator Bluff Detection Telemetry & Activation Conditions

The mediator tracks agent behavior via a rolling window of $W=3$ rounds on `(satisfaction_score, concession_willingness)` tuples per agent. The detector triggers when an agent exhibits persistently low satisfaction ($\bar{s} < 4.0$) and minimal concession ($\bar{c} < 0.2$) across 3 consecutive rounds.

> ℹ️ **Empirical Status:**  
> Because all 30 empirical trials in the Stage a benchmark concluded in round 1 upon the mediator's initial proposal, the rolling 3-round window never accumulated enough history to trigger. The bluff detector is implemented in `src/agents/mediator.py` and unit-tested in `tests/test_strategic.py`, but has no empirical trigger data from live multi-round sessions yet.

### 4 · Benchmark Evaluation with 95% Bootstrap CIs & Wilcoxon Tests

All numbers below are generated directly from the tracked log files (`results/benchmark_business_deal.jsonl` and `results/reservation_sweep.jsonl`) via `scripts/update_readme_results.py`.

#### Stage a: Fixed Benchmark (`business_deal`, `single_text`)

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

> ⚠️ **What This Benchmark Measures (Honest Caveat):**  
> All 30 Consensus Engine trials ended in round 1 with exactly 4 LLM calls each (1 mediator proposal + 3 stakeholder critiques). Every party accepted the mediator's first proposal immediately. The engine's modest edge over the public midpoint (distance to frontier 0.0654 vs 0.0725, Wilcoxon $p = 0.0017^{**}$; Nash welfare $p = 0.0015^{**}$) therefore reflects the mediator adjusting the midpoint using issue descriptions, **NOT iterative negotiation**. This should not be described or interpreted as negotiation superiority. Results reflect one scenario (`business_deal`) and one model config (`20b_vs_20b`).
>
> In symmetric continuous domains without private reservation boundaries, arithmetic baselines like `naive_average` achieve high utilitarian welfare (0.9525) because the arithmetic mean is an effective centroid when preferences are symmetric. Note that `private_ideal_average` is an alias for `naive_average` (evaluated across 4 baselines, N=120 baseline trials; N=150 total across all methods).

#### Stage d: Reservation Impasse & Strategic Sweeps

Reproduction command:
```bash
python scripts/sweep_anchor_and_reservation.py --n_engine 30 --n_baselines 100 --model_config 20b_vs_20b
```

##### 1. Reservation Value Sweep (Failure Rate & Breach Rate)

> **Definition of Breach**: An outcome where an agreement was reached, but at least one agent's *true* utility fell below its private reservation value ($u_i < r_i$). Under individual rationality, a rational negotiator must exit into safe impasse rather than ratify a sub-reservation agreement.

<!-- RESULTS:START reservation_sweep_table -->
| Reservation ($r$) | Midpoint Failure% ($N=100$) | IR-gated Midpoint Agree% ($N=100$) | Naive Breach% ($N=100$) | Engine Agreement Rate | Engine Involuntary Breach Rate |
|---|---|---|---|---|---|
| $r = 0.40$ | 0.0% | 100.0% | 0.0% | 100.0% (2/2 trials) | 0.0% by construction (enforced by `would_accept`, covered by `tests/test_individual_rationality.py`) |
| $r = 0.55$ | 4.0% | 96.0% | 0.0% | *not yet run (resume with `python scripts/sweep_anchor_and_reservation.py`)* | 0.0% by construction (enforced by `would_accept`, covered by `tests/test_individual_rationality.py`) |
| $r = 0.60$ | 33.0% | 67.0% | 1.0% | *not yet run (resume with `python scripts/sweep_anchor_and_reservation.py`)* | 0.0% by construction (enforced by `would_accept`, covered by `tests/test_individual_rationality.py`) |
| $r = 0.65$ | 78.0% | 22.0% | 24.0% | *not yet run (resume with `python scripts/sweep_anchor_and_reservation.py`)* | 0.0% by construction (enforced by `would_accept`, covered by `tests/test_individual_rationality.py`) |
| $r = 0.70$ | 100.0% | 0.0% | 67.0% | *not yet run (resume with `python scripts/sweep_anchor_and_reservation.py`)* | 0.0% by construction (enforced by `would_accept`, covered by `tests/test_individual_rationality.py`) |
<!-- RESULTS:END reservation_sweep_table -->

*Key Takeaway*: Involuntary breach rate for Consensus Engine (and any IR-gated method, including IR-gated midpoint) is **0.0% by construction**: Individual Rationality is strictly enforced in code (`would_accept()` checks $u_i \ge r_i$, tested in `tests/test_individual_rationality.py`). Because any gated method rejects sub-reservation proposals by design, agreement rate under tightening reservation values ($r \ge 0.55$) is the primary empirical metric.

##### 2. Strategic Anchor Inflation Sweep ($N=100$ baselines)

<!-- RESULTS:START anchor_sweep_table -->
*Deterministic baseline arithmetic, no LLM*

| Inflation Level | Stated Price Shift ($\Delta P$) | Naive Baseline Price Shift | Naive Capture Rate | Midpoint Baseline Shift |
|---|---|---|---|---|
| **+10%** | +$9.00 | +$3.00 ± $0.00 | 3.3% | +$0.00 (0.0% capture) |
| **+25%** | +$22.50 | +$7.50 ± $0.00 | 8.3% | +$0.00 (0.0% capture) |
| **+50%** | +$45.00 | +$15.00 ± $0.00 | 16.7% | +$0.00 (0.0% capture) |
| **+75%** | +$67.50 | +$22.50 ± $0.00 | 25.0% | +$0.00 (0.0% capture) |
<!-- RESULTS:END anchor_sweep_table -->

*Key Takeaway*: Naive coordinate averaging is linearly exploited by unconstrained anchor inflation, conceding up to 25.0% of the entire price range. Midpoint coordinate averaging resists price shifts because it only considers domain bounds $[10, 100]$, but cannot express cross-issue trade-offs.

### 5 · Auditable Streaming Trial Log (`results/*.jsonl`)

Every trial emits an unedited JSONL record containing raw inputs, proposals, utility evaluations, and fairness metrics for total research reproducibility:

<!-- RESULTS:START sample_record -->
```json
{"trial_id": 42, "scenario_name": "business_deal", "seed": 42, "method": "consensus_engine", "protocol": "single_text", "agreement_reached": true, "rounds_taken": 1, "per_agent_utilities": {"SupplierCo": 0.5754, "BuyerInc": 0.5793, "LogiTrans": 0.8753}, "final_proposal": {"unit_price": 55.0, "order_volume": 5000.0, "delivery_days": 15.0, "payment_terms": 45.0, "quality_tier": 3.0}, "pareto": {"distance_to_frontier": 0.06, "efficiency_ratio": 0.906, "utilitarian_welfare_ratio": 0.906, "is_pareto_optimal": false, "social_welfare": 2.03, "near_frontier": false}, "fairness": {"nash_welfare": 0.2918, "min_utility": 0.5754, "max_utility": 0.8753, "gini_coefficient": 0.0985, "envy_free": false, "utility_spread": 0.2999}, "privacy": null, "honesty_levels": {"SupplierCo": 1.0, "BuyerInc": 1.0, "LogiTrans": 1.0}, "model_config_name": "120b_vs_20b", "model_names": {"negotiator_model": "openai/gpt-oss-120b", "mediator_model": "openai/gpt-oss-20b"}, "git_commit_hash": "a8328e8", "timestamp": "2026-11-01T17:28:31.651520+00:00", "bluff_detection_scores": {"SupplierCo": {"avg_satisfaction": 6.5, "avg_concession": 0.6, "rounds": 1.0}, "BuyerInc": {"avg_satisfaction": 3.2, "avg_concession": 0.6, "rounds": 1.0}, "LogiTrans": {"avg_satisfaction": 7.5, "avg_concession": 0.4, "rounds": 1.0}}}
```
<!-- RESULTS:END sample_record -->

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+
- [Groq API Key](https://console.groq.com) (free tier works)

### Install

```bash
git clone https://github.com/Saksham-19-cyber/Consensus-Engine.git
cd Consensus-Engine
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
```

### Run Tests (no API key needed)

```bash
pytest -v
# Expected: 132 passed (+ 1 live integration test requiring GROQ_API_KEY)
```

### Run a Negotiation *(requires GROQ_API_KEY)*

```python
from scenarios.business_deal import BusinessDealScenario
from src.protocol.graph import run_negotiation

scenario = BusinessDealScenario()
profiles, issues = scenario.generate(seed=42)

# Standard mediated negotiation
outcome = run_negotiation(
    profiles=profiles,
    issues=issues,
    scenario_name="business_deal",   # enables ChromaDB memory
    protocol="single_text",
    max_rounds=10,
)

print(f"Agreed: {outcome.agreement_reached}")
print(f"Rounds: {outcome.rounds_taken}")
print(f"Utilities: {outcome.per_agent_utilities}")
print(f"Protocol: {outcome.protocol_used}")
```

### Run with Strategic Agents *(requires GROQ_API_KEY)*

```python
from scenarios.strategic_negotiation import StrategicNegotiationScenario
from src.protocol.graph import run_negotiation
from src.eval.privacy import measure_privacy_leakage

scenario = StrategicNegotiationScenario()

# Mixed-honesty population: honesty_level ~ Uniform(0.3, 1.0)
profiles, issues = scenario.generate(seed=42)
for p in profiles:
    print(f"  {p.name}: honesty_level={p.honesty_level:.2f} (strategic={p.is_strategic})")

outcome = run_negotiation(
    profiles=profiles,
    issues=issues,
    scenario_name="strategic_negotiation",
)

# Measure how much the transcript leaked about each agent's preferences
privacy = measure_privacy_leakage(
    transcript=outcome.messages,
    profiles=profiles,
    issues=issues,
)
cos_str = f"{privacy.mean_cosine_similarity:.3f}" if privacy.mean_cosine_similarity is not None else "N/A"
u_str = f"{privacy.uniform_baseline_cosine:.3f}" if privacy.uniform_baseline_cosine is not None else "N/A"
d_str = f"{privacy.dirichlet_baseline_cosine:.3f}" if privacy.dirichlet_baseline_cosine is not None else "N/A"
lift_str = f"{privacy.lift_over_uniform:.3f}" if privacy.lift_over_uniform is not None else "N/A"
print(f"Mean privacy leakage (cosine): {cos_str}")
print(f"  Uniform-guess baseline: {u_str}")
print(f"  Dirichlet baseline: {d_str}")
print(f"  Lift over uniform: {lift_str}")
print(f"Bluff suspects: {outcome.bluff_detection_scores}")
```

### Launch the Full Application

```bash
# Terminal 1: FastAPI backend
python -m src.api.main
# → http://localhost:8000/docs

# Terminal 2: Next.js research web frontend
cd web && npm run dev
# → http://localhost:3000

# Terminal 3 (Optional): Streamlit dashboard
streamlit run frontend/app.py
# → http://localhost:8501
```

---

## 📡 API Reference

> [!NOTE]
> **Public Endpoint Limits & Rate Limiting**:
> To protect compute resources, endpoints calling LLM inference enforce:
> - **Per-IP Rate Limit**: 10 requests per minute per client IP across `POST /api/negotiate`, `POST /api/negotiate/start`, `POST /api/scenario/parse`, and `POST /api/eval/run` (returns `429 Too Many Requests`).
> - **Negotiation Cap**: `max_rounds` capped at $\le 10$ (returns `400 Bad Request` if exceeded).
> - **Evaluation Cap**: `n_trials` on public batch runs capped at $\le 5$ (returns `400 Bad Request` if exceeded).

### `POST /api/negotiate` or `/api/negotiate/start`

Initiate a negotiation session. Supports both named scenarios and pre-parsed custom scenarios.

**Request body** (`NegotiateRequest`):
```json
{
  "scenario": "business_deal",
  "n_agents": 3,
  "max_rounds": 10,
  "seed": 42,
  "protocol": "single_text",
  "model_config": "20b_vs_20b",
  "parsed_scenario_token": null
}
```

* Query Parameter: `sync=true` returns the complete `SessionResponse` synchronously; `sync=false` (default) dispatches a background task.

**Synchronous Response** — `SessionResponse` *(schema example)*:
```json
{
  "session_id": "a1b2c3d4",
  "scenario": "business_deal",
  "status": "completed",
  "outcome": {
    "final_proposal": {"unit_price": 55.0, "order_volume": 5000.0, "delivery_days": 15.0, "payment_terms": 45.0, "quality_tier": 3.0},
    "per_agent_utilities": {"SupplierCo": 0.58, "BuyerInc": 0.58, "LogiTrans": 0.88},
    "agreement_reached": true,
    "rounds_taken": 1,
    "protocol_used": "single_text",
    "bluff_detection_scores": {}
  },
  "messages": [
    {"round_number": 1, "agent_name": "Mediator", "role": "mediator", "content": "Initial proposal...", "message_type": "proposal"}
  ]
}
```

### `GET /api/negotiate/{job_id}` or `/api/negotiate/job/{job_id}`

Poll status and live streaming messages of an active or completed background negotiation job.

**Response** — `JobStatusResponse` *(schema example)*:
```json
{
  "job_id": "a1b2c3d4",
  "session_id": "a1b2c3d4",
  "status": "completed",
  "scenario": "business_deal",
  "messages": [],
  "outcome": { "agreement_reached": true, "rounds_taken": 1 },
  "summary": null,
  "error": null
}
```

### `POST /api/scenario/update`

Re-validates user-edited issues and stakeholder profiles server-side and re-checks mathematical feasibility.

**Request body** (`UpdateScenarioRequest`):
```json
{
  "issues": [
    {"name": "salary", "range": [50000.0, 150000.0], "description": "Annual base salary"}
  ],
  "stakeholders": [
    {
      "name": "Candidate",
      "role": "Engineer",
      "persona": "",
      "weights": {"salary": 1.0},
      "ideal_values": {"salary": 140000.0},
      "reservation_value": 0.50
    }
  ]
}
```

**Response**: Returns updated `ParseScenarioResponse` with new `parsed_scenario_token`.

### `POST /api/eval/run`

Run a benchmark batch with bootstrap statistical aggregation across methods.

```json
{
  "scenario": "business_deal",
  "n_trials": 30,
  "methods": ["naive_average", "public_midpoint", "nash_bargaining", "consensus_engine"],
  "model_config": "20b_vs_20b",
  "protocol": "single_text",
  "run_privacy_probe": false,
  "seed": 42
}
```

### `GET /api/eval/report/{scenario}`

Returns the latest aggregated Markdown evaluation report for a scenario, searching `results/` first, falling back to `data/logs/`.

---

## 📁 Scenarios

| Scenario | Parties | Issues | Key Tension |
|---|---|---|---|
| `roommate` | 2 | Rent split, cleaning, noise, guests | Distributive |
| `business_deal` | 3 | Price, volume, delivery, payment, quality | Integrative |
| `trip_planning` | 3–5 | Budget, destination, dates, hotel, activities | Coalition |
| `strategic_negotiation` | 3 | Price, volume, delivery, payment, quality | Strategic deception |

All scenarios use `np.random.RandomState(seed)` for full reproducibility. Each call to `generate(seed=i)` produces a distinct but deterministic agent population.

---

## 🧩 Extending the System

### Add a New Scenario

```python
# scenarios/my_scenario.py
from scenarios.base import Scenario
from src.models.utility import Issue, UtilityFunction, StakeholderProfile
import numpy as np

class MyScenario(Scenario):
    name = "my_scenario"
    ISSUES = [
        Issue(name="issue_a", min_value=0.0, max_value=100.0),
        Issue(name="issue_b", min_value=0.0, max_value=50.0),
    ]

    def generate(self, seed=42):
        rng = np.random.RandomState(seed)
        # ... build profiles
        return profiles, self.get_issues_meta(self.ISSUES)
```

### Add a Strategic Agent Population

```python
# All agents strategic with honesty_level=0.4
profiles, issues = scenario.generate_all_strategic(seed=42, honesty=0.4)

# Mixed: one fully honest, two strategic
profiles[0] = profiles[0].model_copy(update={"honesty_level": 1.0, "strategic_bias": {}})
```

### Add a New Baseline

```python
# src/eval/baselines.py — add to the dispatch dict
def compute_baseline(method: str, profiles, issues) -> dict[str, float]:
    ...
    elif method == "my_baseline":
        return my_custom_logic(profiles, issues)
```

---

## 🏛 Design Decisions

**Why Groq?** High-speed LLM inference. A single mediation round requires 3–5 structured LLM calls in sequence (mediator proposal + stakeholder critiques). Fast inference keeps round turnaround responsive across multi-agent loops.

**Why LangGraph?** The negotiation DAG has non-trivial conditional branching (agree / impasse / revise / issue-link). LangGraph's typed state and conditional edges make the flow explicit and testable rather than buried in a callback chain.

**Why is `acceptable` always true-utility?** Strategic agents should not be able to bluff the system into perpetual impasse. The separation between *stated* utility (used in dialogue) and *true* utility (used for termination decisions) prevents degenerate negotiations.

**Why bootstrap CI over t-test?** Utility distributions are bounded, often skewed, and small-sample. Bootstrap CI makes no normality assumption. Wilcoxon ranks are used for hypothesis tests for the same reason.

---

## 🖥 Frontend Interfaces & Deployment Links

Consensus Engine provides two dedicated frontend interfaces for exploring multi-agent negotiations and empirical benchmarks:

| Interface | Technology | Primary Purpose | Live URL / Local Access |
|---|---|---|---|
| **Research Web App** | Next.js 16.3.4, TypeScript, Tailwind, Recharts | Interactive runner, 95% bootstrap CI charts, privacy probe simplex visualizer, JSONL trial explorer | **[consensus-engine-opal.vercel.app](https://consensus-engine-opal.vercel.app/)** <br> *(Local: `http://localhost:3000`)* |
| **Streamlit Analytics Dashboard** | Streamlit, Python, Requests | Rapid local experimentation, session playback, parameter sweeps | **[consensus-engine.streamlit.app](https://consensus-engine-y9ybfxmfwcmopgobfzzpjx.streamlit.app/)** <br> *(Local: `http://localhost:8501`)* |
| **FastAPI Orchestration Backend** | FastAPI, LangGraph, ChromaDB, Groq | Agent graph execution, SQLite audit persistence, precedent memory | **[API Docs](http://localhost:8000/docs)** <br> *(Local: `http://localhost:8000`)* |

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FSaksham-19-cyber%2FConsensus-Engine&root-directory=web)

---

### 1. Next.js Research Web App (`/web`)

* **Live Negotiation Runner**: Configure scenarios (`business_deal`, `roommate`, `strategic_negotiation`), select protocols (`single_text`, `alternating_offers`), and inspect round-by-round proposals, agent critiques, and real-time mediator bluff detection flags.
* **Benchmark Dashboard**: Interactive bar charts with **95% bootstrap confidence interval error bars**, Wilcoxon significance badges ($p < 0.05^*$, $p < 0.01^{**}$, n.s.), reservation breach curves ($N=100$), and strategic anchor manipulation span capture comparisons ($N=30$).
* **Privacy Probe View**: Visualizes per-agent preference leakage against uniform and Dirichlet baselines with interactive weight simplex radar charts.
* **Auditable Trial Explorer**: Browse individual streaming JSONL trial records from `results/` (or `data/logs/`) with expandable round dialogues and raw JSON inspect/copy modals.

#### Running Next.js Locally:
```bash
cd web
npm install
npm run dev
# Open http://localhost:3000
```

---

### 2. Streamlit Dashboard (`/frontend`)

> **Note on Frontend Differences:** The Streamlit dashboard is the project's legacy/rapid-experimentation interface, dedicated strictly to running and inspecting the 4 fixed benchmark scenarios (`roommate`, `business_deal`, `trip_planning`, `strategic_negotiation`). The **Free-Form Scenario Builder** is implemented **exclusively in the Next.js application (`/web`)**.

```bash
# Start backend orchestrator
uvicorn src.api.main:app --port 8000 &

# Start Streamlit frontend
streamlit run frontend/app.py
# Open http://localhost:8501
```

---

### 🌐 Deployment Architecture (Vercel)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Vercel Deployment                               │
│                                                                        │
│  Next.js 16.3.4 App Router (Frontend)   FastAPI Serverless Function   │
│  - Recharts Empirical Dashboards        - api/index.py (ASGI app)      │
│  - Privacy Simplex Visualizer           - includeFiles: "results/**"   │
│  - JSONL Trial Record Explorer          - GET /api/logs, /api/eval/... │
└────────────────────────────────────────────────────────────────────────┘
```

#### Frontend & Backend Deployment on Vercel
Both the Next.js frontend and the FastAPI backend are deployed on Vercel:
1. Root `vercel.json` configures serverless routing:
   * `/api/(.*)` rewrites to `api/index.py` (which mounts the FastAPI ASGI app from `src.api.main:app`).
   * `functions: { "api/index.py": { "includeFiles": "results/**", "maxDuration": 300 } }` ensures tracked empirical result logs in `results/` are bundled into the serverless function.
2. In Vercel Project Settings → **Environment Variables**, set:
   ```env
   GROQ_API_KEY=your_groq_api_key
   ```
   For client-side API routing during local development or detached mode, set `NEXT_PUBLIC_API_URL` to your backend origin (or leave empty to use relative `/api` rewrites).

#### Persistence & Serverless Function Limitations
* **Serverless Filesystem:** Vercel serverless function execution is ephemeral; files outside `/tmp` are read-only. Tracked benchmark results are bundled statically via `functions.includeFiles`, while SQLite logs in ephemeral runtime are temporary unless configured with external database backends.
* **Execution Duration Limits:** Under Vercel's Fluid Compute architecture, the default and standard maximum timeout for the Hobby (Free) tier is **300 seconds (5 minutes)**. On the Pro tier, standard maximum execution is **800 seconds (~13.3 minutes)**, with extended execution up to **1,800 seconds (30 minutes)** supported in beta for Python runtimes (see [Vercel Function Duration Configuration](https://vercel.com/docs/functions/configuring-functions/duration)).

---

## 🧪 Free-Form Scenario Builder (Exploratory Mode)

> **Frontend Availability:** The Free-Form Scenario Builder is implemented **exclusively in the Next.js web application (`/web`)** under the **Scenario Builder** tab. It is not available in the legacy Streamlit dashboard (`frontend/app.py`), which remains dedicated to the 4 fixed benchmark scenarios.
>
> **Scope notice:** Free-form scenarios are **not** covered by the N=30 statistical benchmarks documented in this README. The Pareto efficiency ratios, agreement rates, and Wilcoxon test results in the benchmark tables apply exclusively to the four fixed scenarios (`business_deal`, `roommate`, `trip_planning`, `strategic_negotiation`). Results from free-form scenarios are illustrative and exploratory — they should not be cited as evidence of the engine's negotiation performance.

Consensus Engine includes a free-form scenario input mode that allows you to describe a negotiation situation in plain English. The LLM parses your description into structured `StakeholderProfile` and `Issue` objects, which are then passed directly into the existing negotiation engine — no modifications to the core protocol are required.

### Design Model

This feature implements **User-Describes-All-Parties** mode with **LLM-Inferred Fallback**:

- The user describes the negotiation situation and, optionally, each party's priorities or constraints.
- Any party whose preferences are **not** fully specified by the user gets LLM-generated ("LLM-inferred") utility weights, ideal values, and reservation values.
- The returned data structure explicitly tags each stakeholder's source as `user_specified` or `llm_inferred`. The UI displays this distinction prominently — invented numbers are never presented as user-provided data.

**This is a simulation and design tool, not a live multi-party negotiation platform.** All parties' preferences are either user-supplied or model-generated. The system does not connect to or represent any real third parties.

### CLI Execution
You can parse and execute custom scenarios directly via the CLI:
```bash
python -m scenarios.from_text --text "FounderCo is licensing software to EnterpriseCorp. Issues: annual license fee ($50,000 to $500,000), contract duration (1 to 5 years), and support tier (1=basic, 2=standard, 3=dedicated)." --rounds 3 --protocol single_text
```

### Flow & Core Invariants

```
POST /api/scenario/parse   →  Confirm & Edit (UI / CLI)  →  Feasibility Check  →  POST /api/negotiate
      (parse only)             (user inspects & modifies)       (u_i >= r_i for all)       (with token)
```

1. **Describe** — Write a plain-language description of the negotiation in the Next.js **Scenario Builder** tab or CLI.
2. **Parse & Ordinal-to-Cardinal Mapping** — The LLM extracts issues with numeric ranges and maps ordinal tiers (e.g., support tiers: 1=basic, 2=standard, 3=dedicated) to cardinal intervals.
3. **Confirm & Edit Step** — The UI presents parsed issues, ranges, party ideal values, weights, and reservation thresholds with clear provenance tags (`user_specified` vs `llm_inferred`). **The user can edit any weight or issue range directly** before running.
4. **Pre-Flight Feasibility Check** — Before invoking the negotiation graph, the engine tests whether a mutually acceptable deal is mathematically possible: checking whether any point in the issue space gives every party $U_i \ge r_i$ (via exhaustive grid search for $\le 4$ issues, or Monte Carlo sampling for 5–6 issues). If infeasible, an amber warning is raised immediately, avoiding wasted LLM calls.
5. **Run & Plain-Language Summary** — The confirmed scenario executes under `single_text` or `alternating_offers`, outputting both raw numbers and an accessible plain-language narrative.

### Empirical Validation: 4 Live Custom Checks (N=3 + 1 Infeasible)

Documented in [`RESULTS.md`](RESULTS.md#stage-f-custom-scenario-checks-illustrative-n3-not-a-benchmark):
1. **Software Licensing (3 issues, 2 parties)**: Feasible. Agreement reached in 1 round at Annual License Fee $260k, Duration 3 yrs, Support Tier 2 (distance to frontier: 0.1000).
2. **Salary Negotiation (3 issues, 2 fully-specified parties)**: Feasible. Agreement reached in 1 round at Salary $165k, Equity 6,000 shares, 45 remote days (distance to frontier: 0.0259).
3. **Urban Development (3 issues, 3 parties with 1 LLM-inferred)**: Feasible. Agreement reached in 1 round at Height 17.5 stories, Green space 27.5%, Affordable units 55 (distance to frontier: 0.0212).
4. **Deliberately Infeasible Dispute (Reservation 0.95 each)**: Pre-flight feasibility check detected negative joint surplus (-0.450) and flagged an immediate warning. Invariant verification confirmed all weight vectors sum strictly to 1.0 and $U(\text{ideal}) = 1.0$.

### Issue Count Limits

| Issue count | Pareto computation | Result label | Method Note |
|---|---|---|---|
| ≤ 4 | Exhaustive grid search ($12^N$ combos) | Exact | Ground truth discrete frontier |
| 5–6 | Monte Carlo sampling (100,000 random points) | **Approximate** | Approximate utilitarian welfare ratio for high-dimensional spaces |
| > 6 | Rejected (400 error) | — | Exceeds dimensionality limit |

The grid search in `src/eval/pareto.py` is exponential in issue count ($12^N$ combos at resolution 12; 248,832 points at 5 issues, ~3M at 6). For 5–6 issues the system switches to `monte_carlo_pareto_frontier()` and labels results `approximate` in the API response and UI. Monte Carlo sampling over $[min, max]^N$ evaluates 100,000 randomly sampled packages to construct a non-dominated candidate frontier. In reports and API responses, metrics derived from Monte Carlo frontiers are explicitly flagged with `is_approximate_frontier: true`.

### API

#### `POST /api/scenario/parse`

Parse a description into structured scenario objects. Does **not** run a negotiation.

```json
// Request
{ "description": "A landlord and tenant are negotiating...", "seed": 42 }

// Response
{
  "issues": [{ "name": "rent", "min_value": 800, "max_value": 1500, "description": "..." }],
  "stakeholders": [
    { "name": "Landlord", "source": "user_specified", "weights": {...}, ... },
    { "name": "Tenant",   "source": "llm_inferred",   "weights": {...}, ... }
  ],
  "field_notes": ["⚠️  Tenant's preferences were generated by the LLM..."],
  "warnings": [],
  "pareto_mode": "exhaustive",
  "issue_count": 3,
  "parsed_scenario_token": "<opaque base64 token>"
}
```

#### `POST /api/negotiate` (extended)

Pass the token from the parse step to run the confirmed scenario:

```json
{
  "scenario": "free_form",
  "max_rounds": 5,
  "parsed_scenario_token": "<token from parse response>"
}
```

When `parsed_scenario_token` is present, the endpoint decodes the pre-parsed `(profiles, issues)` tuple and calls `run_negotiation()` directly — no named scenario lookup occurs.

### Security & Token Trust Model

The `parsed_scenario_token` round-tripped between `/api/scenario/parse` and `/api/negotiate` is an opaque base64-encoded JSON blob:
- **Data transparency & privacy:** The token serializes only structured schema objects (`issues_meta` and Pydantic-exported `StakeholderProfile` dicts). It **does not** contain raw user description text, internal LLM reasoning/scratchpad traces, or user PII.
- **Trust model & known limitation:** The token is currently **unsigned base64 JSON**. On decode, `ParseReport.decode_token()` reconstructs and validates the objects through Pydantic (ensuring schema correctness, type safety, and normalized weights), but does **not** cryptographically verify server-side authenticity (no HMAC). A client could tamper with utility weights, reservation values, or issue ranges before calling `POST /api/negotiate`.
- **Deployment recommendation:** For this exploratory demo and research simulator with no authentication layer, this is an accepted trade-off. If deploying in a multi-tenant or production environment with untrusted clients, wrap the token payload in an HMAC signature (e.g., using `itsdangerous.URLSafeSerializer`).

### Frontend

Navigate to the **Scenario Builder** tab in the web UI. The amber banner is shown on every screen:

> *"Free-form scenarios are not covered by this project's N=30 statistical benchmarks; results here are illustrative, not validated."*

Party cards display a **green "User-specified"** badge or an **amber "LLM-inferred"** badge, making provenance visible at a glance before you confirm and run.

---

## ⚠️ Limitations & Honest Research Disclosures

1. **Mediator Midpoint Adjustment vs. Iterative Negotiation**:
   - In the Stage a benchmark (`business_deal`, $N=30$), all 30 Consensus Engine trials concluded in round 1 with exactly 4 LLM calls each (1 mediator proposal + 3 stakeholder critiques). Every party accepted the mediator's initial proposal.
   - The engine's slight advantage over the public midpoint (distance to frontier 0.0654 vs 0.0725, $p=0.0017^{**}$; Nash welfare 0.3990 vs 0.3885, $p=0.0015^{**}$) therefore reflects the mediator adjusting the midpoint using issue descriptions, **NOT iterative back-and-forth negotiation**. In symmetric continuous domains without tight private reservations, arithmetic baselines like `naive_average` achieve high utilitarian welfare (0.9525) with zero LLM calls.
2. **Unrun Stages in Benchmark Suite**:
   - Stage a was evaluated across $N=30$ live trials (120 real LLM calls) with 4 baselines ($N=120$ baseline trials, $N=150$ total trials across all methods). Stage d reservation sweep has $N=2$ trials for $r=0.40$ and is unrun for other tiers. Stage b (`alternating_offers`), Stage c (`strategic_negotiation`), and Stage e (privacy probe) are not yet run in batch benchmarks.
3. **Bluff Detector Lacks Empirical Trial Evidence**:
   - The mediator's rolling bluff detector requires $W \ge 3$ consecutive rounds of critique history ($\bar{s}_i < 4.0 \;\wedge\; \bar{c}_i < 0.2$). Because all Stage a benchmark trials concluded in round 1, the bluff detector never triggered in real benchmark runs. While unit-tested in `tests/test_strategic.py`, it currently has no empirical activation data from extended multi-round sessions.
4. **Simulated Agents vs. Human Bargaining**:
   - LLM stakeholder personas simulate bounded concession dynamics and verbal posturing, but cannot fully replicate human psychological factors, reputation concerns, or long-term dynamic relationship considerations.

---

## 📋 Requirements

```
Python        ≥ 3.11
Node.js       ≥ 20.0      # Next.js frontend in /web
groq          ≥ 0.11.0    # LLM provider
langgraph     ≥ 0.2.0     # Negotiation graph orchestration
chromadb      ≥ 0.5.0     # Precedent memory
pydantic      ≥ 2.0       # Schema validation
numpy         ≥ 1.26.0    # Pareto computation, bootstrap
scipy         ≥ 1.13.0    # Wilcoxon tests
fastapi       ≥ 0.115.0   # REST API
next          ≥ 16.0      # Web frontend (Next.js 16.3.4)
recharts      ≥ 2.15.0    # Benchmark visualisations
```

---

## 📄 License

MIT © 2026 — [Saksham-19-cyber](https://github.com/Saksham-19-cyber)

---

<div align="center">

*Built to answer: can LLMs negotiate genuinely, or are they just agreeable?*

</div>
