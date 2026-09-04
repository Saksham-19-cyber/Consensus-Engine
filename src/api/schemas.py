from __future__ import annotations
from pydantic import BaseModel, Field


class NegotiateRequest(BaseModel):
    scenario: str = "roommate"
    n_agents: int = Field(default=2, ge=2, le=5)
    max_rounds: int = Field(default=10, ge=1, le=30)
    seed: int = 42
    # Free-form mode: when present, overrides `scenario` with pre-parsed data.
    # This is the opaque token returned by POST /api/scenario/parse.
    parsed_scenario_token: str | None = None


class EvalRequest(BaseModel):
    scenario: str = "roommate"
    n_trials: int = Field(default=10, ge=1, le=500)
    methods: list[str] = Field(default_factory=lambda: ["naive_average", "nash_bargaining"])
    seed: int = 42
    n_agents: int = Field(default=2, ge=2, le=5)


class SessionResponse(BaseModel):
    session_id: str
    scenario: str
    status: str
    outcome: dict | None = None
    messages: list[dict] = Field(default_factory=list)


class EvalResponse(BaseModel):
    batch_id: str
    scenario: str
    n_trials: int
    summary: dict = Field(default_factory=dict)
    report_markdown: str = ""


# ── Free-form scenario parse endpoint schemas ─────────────────────────────────

class ParseScenarioRequest(BaseModel):
    """Request body for POST /api/scenario/parse."""
    description: str = Field(
        ...,
        min_length=20,
        description="Natural-language description of the negotiation scenario.",
    )
    seed: int = Field(default=42, description="RNG seed (stored in report for traceability).")


class ParsedIssueOut(BaseModel):
    """One negotiation issue as returned by the parser."""
    name: str
    min_value: float
    max_value: float
    description: str


class ParsedStakeholderOut(BaseModel):
    """One negotiating party as returned by the parser."""
    name: str
    role: str
    persona: str
    source: str  # "user_specified" | "llm_inferred"
    weights: dict[str, float]
    ideal_values: dict[str, float]
    reservation_value: float


class ParseScenarioResponse(BaseModel):
    """
    Response from POST /api/scenario/parse.

    The frontend MUST show this to the user for confirmation before calling
    POST /api/negotiate with the parsed_scenario_token.
    """
    issues: list[ParsedIssueOut]
    stakeholders: list[ParsedStakeholderOut]
    field_notes: list[str]   # provenance notes — which fields were LLM-inferred
    warnings: list[str]      # parser warnings about ambiguities
    pareto_mode: str          # "exhaustive" | "monte_carlo"
    issue_count: int
    # Opaque token: send back as parsed_scenario_token in POST /api/negotiate
    parsed_scenario_token: str
