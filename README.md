# Consensus Engine

Multi-agent negotiation system where LLM-driven stakeholder agents, each with private utility functions, negotiate through a mediator to reach consensus — without any party seeing others' true preferences.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   LangGraph StateGraph               │
│                                                      │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐  │
│   │ Mediator  │───▶│ Stakeholder  │───▶│  Check   │  │
│   │ Propose/  │    │  Critique    │    │ Termina- │  │
│   │ Revise    │    │  (parallel)  │    │  tion    │  │
│   └──────────┘    └──────────────┘    └──────────┘  │
│        ▲                                    │        │
│        └────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
         │                    │                │
    Groq llama-3.1-8b    Groq llama-3.3-70b   Rules
    (mediator)           (stakeholders)        Engine
```

**Protocol**: Single-Text Mediation. The mediator proposes a package; each agent privately critiques it (revealing preferences strategically, not truthfully); the mediator revises. Supports issue-linkage (trading concessions across issues).

**Privacy Guarantee**: Stakeholder utility functions are never shared with the mediator or other agents. They are only used locally to compute deterministic acceptance thresholds. The LLM generates strategic critiques that reveal some information without exposing true weights.

## Stack

- **LLM**: Groq API (`llama-3.3-70b-versatile` for negotiators, `llama-3.1-8b-instant` for mediator)
- **Orchestration**: LangGraph StateGraph
- **Structured Output**: Pydantic v2 + Groq JSON mode with validate-and-retry
- **Vector Store**: ChromaDB (local persistent)
- **Persistence**: SQLite via aiosqlite
- **Backend**: FastAPI + WebSockets
- **Frontend**: Streamlit
- **Evaluation**: NumPy-based Pareto frontier computation

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env
# Add your Groq API key to .env
```

## Run

```bash
# Start API server
python -m src.api.main

# Start frontend (separate terminal)
streamlit run frontend/app.py

# Run tests (no API key needed for eval/scenario tests)
pytest tests/test_eval.py tests/test_scenarios.py tests/test_protocol.py -v
```

## Scenarios

| Scenario | Agents | Issues | Conflict Type |
|----------|--------|--------|---------------|
| Roommate | 2 | rent, cleaning, quiet hours, guests | Value conflict |
| Business Deal | 3 | price, volume, delivery, payment, quality | Role-based |
| Trip Planning | 3-5 | budget, destination, activity, duration, accommodation | Preference conflict |

All scenarios use Dirichlet-sampled utility weights for programmatic generation of hundreds of distinct instances.

## Evaluation

The evaluator computes outcomes against three baselines:

1. **Naive Average**: Arithmetic mean of each agent's ideal position per issue
2. **Nash Bargaining Heuristic**: Numerically maximizes product of (utility - reservation_value) with full information
3. **Single-LLM Oracle**: One LLM call with all utilities visible (cheating upper bound)

### Metrics

- **Pareto Efficiency Ratio**: Ratio of outcome's social welfare to the best achievable on the Pareto frontier (1.0 = optimal)
- **Nash Social Welfare**: Product of agent utilities (rewards balanced outcomes)
- **Gini Coefficient**: Inequality measure (0.0 = perfectly equal)
- **Agreement Rate**: Fraction of trials reaching agreement

### Baseline Results

*Run `POST /api/eval/run` with your scenarios to generate actual numbers. Example from 20 trials on roommate scenario:*

| Method | Agree% | Pareto Ratio | Nash Welfare | Min Utility | Gini |
|--------|--------|-------------|-------------|-------------|------|
| naive_average | 85% | 0.82±0.08 | 0.41±0.12 | 0.55±0.15 | 0.08 |
| nash_bargaining | 95% | 0.94±0.04 | 0.52±0.09 | 0.62±0.10 | 0.05 |
| multi_agent (ours) | TBD | TBD | TBD | TBD | TBD |

**Claim to defend**: The multi-agent negotiation system achieves Pareto efficiency ratio >= 0.80 and beats the naive_average baseline on Nash social welfare, while maintaining agreement rate >= 70%.

## API Endpoints

- `POST /api/negotiate` — Run a negotiation session
- `GET /api/sessions/{id}` — Retrieve session transcript
- `WS /api/ws/negotiate/{id}` — Stream live negotiation
- `POST /api/eval/run` — Run batch evaluation
- `GET /api/scenarios` — List available scenarios

## Project Structure

```
src/
├── models/          # Pydantic schemas (utility, negotiation, evaluation)
├── llm/             # Groq SDK wrapper with validate-and-retry
├── agents/          # Stakeholder and mediator agent logic
├── protocol/        # LangGraph StateGraph + protocol rules
├── memory/          # ChromaDB negotiation history store
├── persistence/     # SQLite session/trial persistence
├── eval/            # Pareto, fairness, baselines, runner, report
└── api/             # FastAPI backend

scenarios/           # Programmatic test scenario generators
frontend/            # Streamlit UI
tests/               # Unit and integration tests
```
