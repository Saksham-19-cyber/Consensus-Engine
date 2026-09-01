from __future__ import annotations
from pydantic import BaseModel, Field


class NegotiateRequest(BaseModel):
    scenario: str = "roommate"
    n_agents: int = Field(default=2, ge=2, le=5)
    max_rounds: int = Field(default=10, ge=1, le=30)
    seed: int = 42


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
