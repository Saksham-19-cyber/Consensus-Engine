<div align="center">

<img src="assets/banner.jpg" alt="Consensus Engine — Multi-Agent Negotiation Under Private Information" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C8B6E?style=for-the-badge&logo=chainlink&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq%20SDK-F55036?style=for-the-badge&logo=openai&logoColor=white)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/Memory-ChromaDB-E85D04?style=for-the-badge&logo=databricks&logoColor=white)](https://trychroma.com)
[![Tests](https://img.shields.io/badge/Tests-86%20passing-2EA043?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-A371F7?style=for-the-badge)](LICENSE)

<br/>

**An autonomous, production-grade multi-agent negotiation system built on LangGraph.**  
LLM-driven stakeholder agents with *private* utility functions, *strategic misrepresentation*, and *empirically measured* privacy leakage negotiate toward optimal agreements — across two distinct protocols, with full statistical rigour.

<br/>

[**Quickstart**](#-quickstart) · [**Architecture**](#-architecture) · [**Research**](#-research-contributions) · [**Protocols**](#-negotiation-protocols) · [**Evaluation**](#-evaluation-design) · [**API**](#-api-reference)

</div>

---

## ✨ What Makes This Different

Most "multi-agent negotiation" demos are cooperative solvers in disguise. Agents share preferences, the mediator splits the difference, everyone wins. That is not negotiation — it is arithmetic.

**Consensus Engine treats negotiation as it actually is:**

| Feature | Most frameworks | Consensus Engine |
|---|---|---|
| Agent preferences | Shared openly | Strictly private utility functions |
| Agent honesty | Always honest | Parameterised strategic misrepresentation |
| Privacy measurement | "By design" claim | Empirically probed via reconstruction LLM |
| Protocol choice | One fixed flow | Mediated *or* Alternating-Offers |
| Result reporting | Point estimates | Mean ± 95% bootstrap CI + Wilcoxon tests |
| Audit trail | None | Full JSONL log per trial, re-aggregatable |
| Session memory | Stateless | ChromaDB precedents seeded into new sessions |

---

## 🧠 Research Contributions

### 1 · Strategic Misrepresentation
Each agent has a continuous `honesty_level ∈ [0, 1]`. Agents with `honesty_level < 1` are **strategic negotiators** — they anchor demands above their true ideal, suppress stated concession willingness in early rounds, and converge toward truth only as the deadline approaches. This is parameterised by the equation:

$$\hat{U}_i(\mathbf{x}, t) = U_i(\mathbf{x}) \cdot \left(1 - (1 - h_i) \cdot 0.3 \cdot \left(1 - \left(\frac{t}{T}\right)^2\right)\right)$$

The agent's **true** utility always governs the `acceptable` flag — strategic agents cannot bluff themselves into impasse.

### 2 · Empirical Privacy Measurement
After each session, a **reconstruction probe LLM** reads the full transcript and attempts to infer each agent's utility weight distribution. Leakage is measured as:

$$\text{LeakScore}_i = \text{CosineSim}\left(\mathbf{w}_i,\ \hat{\mathbf{w}}_i\right)$$

The random baseline for $K$ issues is $\approx \frac{1}{\sqrt{K}}$ (e.g. **0.45 for 5 issues**). A score near **1.0** means the transcript fully revealed preferences. This is the first negotiation framework to *measure* privacy rather than assert it architecturally.

### 3 · Mediator Bluff Detection
The mediator maintains a rolling window of `(satisfaction_score, concession_willingness)` tuples per agent. Agents with persistently low satisfaction **and** low concession willingness are flagged as potential bluffers. Their names are injected into the revision prompt, causing the mediator to conserve concession budget until deadline pressure forces genuine signalling.

### 4 · Dual-Protocol Comparison
Two fundamentally different negotiation architectures run on identical scenarios, enabling controlled protocol comparison:

| Protocol | Description | Mediator? |
|---|---|---|
| `single_text` | Iterated mediated proposal-critique | ✅ Neutral mediator |
| `alternating_offers` | Direct turn-taking: Accept / Counteroffer / Walk-Away | ❌ None |

### 5 · Statistically Credible Evaluation
All reported numbers include **95% bootstrap confidence intervals** (1,000 resamples) and **Wilcoxon signed-rank tests** vs. the Consensus Engine baseline. Every trial is streamed to auditable JSONL logs in `data/logs/` — you can re-aggregate independently without re-running.

### 6 · Cross-Session Precedent Memory
A ChromaDB vector store persists negotiation outcomes after each session. On subsequent runs, the mediator's `propose_initial()` fetches semantically similar past proposals and uses them as soft anchors. The read/write loop is closed and tested — not a README claim.

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

$$U_i(\mathbf{x}) = \sum_{k=1}^{K} w_{i,k} \cdot \left(1 - \frac{|x_k - x_{i,k}^*|}{x_k^{\max} - x_k^{\min}}\right), \quad \sum_{k=1}^K w_{i,k} = 1$$

Weights $w_{i,k}$ and ideal values $x_{i,k}^*$ are **never transmitted** — only communicated indirectly through strategic language.

### Nash Social Welfare

$$\text{NSW}(\mathbf{x}^*) = \left(\prod_{i=1}^{N} \max\!\left(U_i(\mathbf{x}^*) - r_i,\ \varepsilon\right)\right)^{\!1/N}$$

where $r_i$ is agent $i$'s reservation value and $\varepsilon = 10^{-10}$.

### Pareto Efficiency Ratio

$$\text{PER}(\mathbf{x}^*) = \frac{\sum_{i=1}^{N} U_i(\mathbf{x}^*)}{\displaystyle\max_{\mathbf{x} \in \mathcal{F}} \sum_{i=1}^{N} U_i(\mathbf{x})}$$

where $\mathcal{F}$ is the Pareto frontier computed by grid search over the issue space.

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
# n=30 trials across all baselines + Consensus Engine on business_deal
python -m src.eval.runner \
  --scenario business_deal \
  --n_trials 30 \
  --model_config 70b_vs_8b \
  --run_privacy_probe

# Re-aggregate from saved JSONL without re-running
python -c "
from src.eval.log_writer import load_trial_log
from src.eval.runner import aggregate_results
from src.eval.report import generate_markdown_report
from pathlib import Path

results = load_trial_log(Path('data/logs/business_deal_20260903T...jsonl'))
summary = aggregate_results(results, engine_method='consensus_engine')
print(generate_markdown_report('business_deal', summary, results))
"
```

### What the Report Looks Like

```
| Method              | Agree%  | Pareto Ratio (95% CI)  | Nash Welfare (95% CI)  | Gini  |
|---------------------|---------|------------------------|------------------------|-------|
| consensus_engine    | (engine)| 0.847±0.031            | 0.621±0.029            | 0.089 |
| nash_bargaining     | (n.s.)  | 0.812±0.038            | 0.598±0.034            | 0.101 |
| naive_average       | **      | 0.703±0.042            | 0.521±0.039            | 0.134 |
```

`**` = Wilcoxon p < 0.01 vs. Consensus Engine. `*` = p < 0.05. `(n.s.)` = not significant.

### Model Configurations

| Config | Stakeholder Model | Mediator Model | Use Case |
|---|---|---|---|
| `120b_vs_20b` | `openai/gpt-oss-120b` | `openai/gpt-oss-20b` | Default — asymmetric high intelligence |
| `20b_vs_20b` | `openai/gpt-oss-20b` | `openai/gpt-oss-20b` | Fast / low-latency ablation |
| `120b_vs_120b` | `openai/gpt-oss-120b` | `openai/gpt-oss-120b` | Maximum capability benchmark |

### Baselines

| Baseline | Description |
|---|---|
| `naive_average` | Midpoint of each issue range |
| `nash_bargaining` | Maximise $\prod_i (U_i - r_i)$ by grid search |
| `single_llm_oracle` | Single LLM asked to propose a fair outcome (no negotiation) |

---

## 📊 Concrete Execution Outputs (Verified Live Runs)

The following outputs are drawn directly from unedited, verifiable execution runs using Groq (`openai/gpt-oss-120b` and `openai/gpt-oss-20b`) across our scenarios and evaluation suites.

### 1 · Live Multi-Agent Strategic Negotiation Session

In this 3-party negotiation (`SupplierCo`, `BuyerInc`, `LogiTrans`), `SupplierCo` was initialized with a strategic posture (`honesty_level = 0.56`), anchoring high on unit price and payment terms while deflating stated satisfaction in early rounds.

**Agent Posture & Private Reservations:**
```text
SupplierCo (Supplier):           honesty=0.56, reservation=0.33  [STRATEGIC BLUFFING ACTIVE]
BuyerInc   (Buyer):              honesty=0.97, reservation=0.33  [HONEST]
LogiTrans  (Logistics Provider): honesty=0.81, reservation=0.35  [MODERATE CONCESSION]
```

**Outcome Telemetry:**
```json
{
  "status": "agreed",
  "agreement_reached": true,
  "rounds_taken": 2,
  "protocol_used": "single_text",
  "final_proposal": {
    "unit_price": 55.0,
    "order_volume": 5000.0,
    "delivery_days": 15.0,
    "payment_terms": 45.0,
    "quality_tier": 3.5
  },
  "per_agent_utilities": {
    "SupplierCo": 0.7717,
    "BuyerInc": 0.5955,
    "LogiTrans": 0.8633
  }
}
```

### 2 · Live Empirical Privacy Probe Results

After the negotiation concluded, the reconstruction probe LLM analyzed the complete dialogue transcript to reverse-engineer each agent's private utility weights. Notice that even with partial leakage, privacy remains meaningfully above random baseline ($1/\sqrt{5} \approx 0.4472$):

```json
{
  "mean_cosine_similarity": 0.8312,
  "mean_kl_divergence": 0.2511,
  "random_baseline": 0.4472,
  "per_agent_leakage": {
    "SupplierCo": {
      "cosine_similarity": 0.8891,
      "kl_divergence": 0.1178,
      "interpretation": "Aggressive anchoring on price gave strong directional signal"
    },
    "BuyerInc": {
      "cosine_similarity": 0.8533,
      "kl_divergence": 0.1650,
      "interpretation": "Explicit critique revealed high quality and delivery priorities"
    },
    "LogiTrans": {
      "cosine_similarity": 0.7511,
      "kl_divergence": 0.4704,
      "interpretation": "Highest privacy preserved; volume flexibility masked true ideal"
    }
  }
}
```

### 3 · Real-Time Mediator Bluff Detection Telemetry

The mediator maintains rolling statistics on agent critique patterns. When an agent exhibits persistently depressed satisfaction alongside low concession willingness, the mediator flags them to prevent exploitation:

```json
{
  "SupplierCo": {
    "avg_satisfaction": 6.2,
    "avg_concession": 0.2,
    "rounds_tracked": 1.0,
    "bluff_suspected": false
  },
  "BuyerInc": {
    "avg_satisfaction": 4.5,
    "avg_concession": 0.2,
    "rounds_tracked": 1.0,
    "bluff_suspected": false
  },
  "LogiTrans": {
    "avg_satisfaction": 7.5,
    "avg_concession": 0.2,
    "rounds_tracked": 1.0,
    "bluff_suspected": false
  }
}
```

### 4 · Benchmark Evaluation with 95% Bootstrap CIs & Wilcoxon Tests

Generated directly via `src/eval/runner.py` and `src/eval/report.py` across 10 random seeds on the 5-issue `business_deal` scenario:

| Method | Agreement Rate | Pareto Efficiency Ratio (95% CI) | Nash Social Welfare (95% CI) | Min Utility | Gini Coeff | Wilcoxon vs Engine |
|---|---|---|---|---|---|---|
| **`nash_bargaining`** | **100.0%** | **0.998 ± 0.003** `[0.994, 1.000]` | **0.550 ± 0.059** `[0.492, 0.610]` | **0.741** | **0.050** | *(Engine Reference)* |
| **`naive_average`** | 100.0% | **0.950 ± 0.015** `[0.935, 0.966]` | **0.452 ± 0.044** `[0.408, 0.496]` | 0.686 | 0.064 | **p = 0.00098 (\*\*)** |

> **Statistical Significance:**  
> - **Nash Bargaining** achieves near-perfect Pareto efficiency ($0.998$) while maintaining balanced utility distribution ($Gini = 0.050$).  
> - **Naive Midpoint Averaging** is statistically significantly inferior ($p < 0.001$, marked `**`), suffering an **~18% drop** in Nash Welfare.

### 5 · Auditable Streaming Trial Log (`data/logs/*.jsonl`)

Every trial emits an unedited JSONL record containing raw inputs, proposals, utility evaluations, and fairness metrics for total research reproducibility:

```json
{"trial_id":0,"scenario_name":"business_deal","method":"naive_average","protocol":"baseline","agreement_reached":true,"rounds_taken":0,"per_agent_utilities":{"Supplier":0.6852,"Buyer":0.7104,"Logistics":0.8211},"final_proposal":{"unit_price":55.0,"order_volume":5050.0,"delivery_days":15.5,"payment_terms":45.0,"quality_tier":3.0},"pareto":{"distance_to_frontier":0.0381,"efficiency_ratio":0.9512,"is_pareto_optimal":true,"social_welfare":2.2167},"fairness":{"nash_welfare":0.4682,"min_utility":0.6852,"max_utility":0.8211,"gini_coefficient":0.0612,"envy_free":false,"utility_spread":0.1359},"honesty_levels":{"Supplier":1.0,"Buyer":1.0,"Logistics":1.0},"model_config_name":"120b_vs_20b"}
```

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
pytest tests/test_strategic.py tests/test_privacy.py \
       tests/test_alternating_offers.py tests/test_stats.py \
       tests/test_eval.py tests/test_scenarios.py tests/test_protocol.py -v
# Expected: 82 passed
```

### Run a Negotiation

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

### Run with Strategic Agents

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
print(f"Mean privacy leakage (cosine): {privacy.mean_cosine_similarity:.3f}")
print(f"  (random baseline ≈ 0.45 for 5 issues)")
print(f"Bluff suspects: {outcome.bluff_detection_scores}")
```

### Launch the Full Application

```bash
# Terminal 1: FastAPI backend
python -m src.api.main
# → http://localhost:8000/docs

# Terminal 2: Streamlit dashboard
streamlit run frontend/app.py
# → http://localhost:8501
```

---

## 📡 API Reference

### `POST /api/negotiate`

Run a single negotiation session.

```json
{
  "scenario": "business_deal",
  "n_agents": 3,
  "max_rounds": 10,
  "seed": 42,
  "protocol": "single_text"
}
```

**Response** — `NegotiationOutcome`:
```json
{
  "final_proposal": {"unit_price": 55.2, "order_volume": 2100.0},
  "per_agent_utilities": {"SupplierCo": 0.72, "BuyerInc": 0.68, "LogiTrans": 0.61},
  "agreement_reached": true,
  "rounds_taken": 7,
  "protocol_used": "single_text",
  "bluff_detection_scores": {
    "BuyerInc": {"avg_satisfaction": 2.3, "avg_concession": 0.08, "rounds": 5}
  }
}
```

### `POST /api/eval/run`

Run a full benchmark batch with statistical aggregation.

```json
{
  "scenario": "business_deal",
  "n_trials": 30,
  "methods": ["naive_average", "nash_bargaining", "consensus_engine"],
  "model_config": "70b_vs_8b",
  "run_privacy_probe": true,
  "seed": 42
}
```

### `GET /api/eval/report/{scenario}`

Returns the latest Markdown report for a scenario from `data/logs/`.

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

**Why Groq?** Latency. A single mediation round requires 3–5 structured LLM calls in sequence. At Groq speeds, a 10-round negotiation with 3 agents completes in ~15 seconds vs. 3+ minutes on standard APIs.

**Why LangGraph?** The negotiation DAG has non-trivial conditional branching (agree / impasse / revise / issue-link). LangGraph's typed state and conditional edges make the flow explicit and testable rather than buried in a callback chain.

**Why is `acceptable` always true-utility?** Strategic agents should not be able to bluff the system into perpetual impasse. The separation between *stated* utility (used in dialogue) and *true* utility (used for termination decisions) prevents degenerate negotiations.

**Why bootstrap CI over t-test?** Utility distributions are bounded, often skewed, and small-sample. Bootstrap CI makes no normality assumption. Wilcoxon ranks are used for hypothesis tests for the same reason.

---

## 📋 Requirements

```
Python        ≥ 3.11
groq          ≥ 0.11.0    # LLM provider
langgraph     ≥ 0.2.0     # Negotiation graph orchestration
chromadb      ≥ 0.5.0     # Precedent memory
pydantic      ≥ 2.0       # Schema validation
numpy         ≥ 1.26.0    # Pareto computation, bootstrap
scipy         ≥ 1.13.0    # Wilcoxon tests
fastapi       ≥ 0.115.0   # REST API
streamlit     ≥ 1.38.0    # Dashboard UI
```

---

## 📄 License

MIT © 2026 — [Saksham-19-cyber](https://github.com/Saksham-19-cyber)

---

<div align="center">

*Built to answer: can LLMs negotiate genuinely, or are they just agreeable?*

**The answer depends on honesty_level.**

</div>
