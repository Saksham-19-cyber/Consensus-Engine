from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class Proposal(BaseModel):
    values: dict[str, float] = Field(default_factory=dict)
    proposed_by: str = "mediator"
    round_number: int = 0

    def to_display(self) -> dict:
        return {k: round(v, 3) for k, v in self.values.items()}


class Critique(BaseModel):
    agent_name: str
    satisfaction_score: float = Field(ge=0.0, le=10.0)
    acceptable: bool = False
    issues_to_improve: list[str] = Field(default_factory=list)
    desired_directions: dict[str, str] = Field(default_factory=dict)
    concession_willingness: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    round_number: int = 0


class MediatorAction(str, Enum):
    PROPOSE = "propose"
    REVISE = "revise"
    LINK_ISSUES = "link_issues"
    DECLARE_IMPASSE = "declare_impasse"
    DECLARE_AGREEMENT = "declare_agreement"


class AgentAction(str, Enum):
    """Actions available in the alternating-offers protocol."""
    ACCEPT = "accept"
    COUNTEROFFER = "counteroffer"
    WALK_AWAY = "walk_away"


class CounterOffer(BaseModel):
    """An agent's direct counter-proposal in the alternating-offers protocol."""
    agent_name: str
    action: AgentAction
    proposal: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""
    round_number: int = 0


class MediatorResponse(BaseModel):
    action: MediatorAction
    revised_proposal: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""
    detected_patterns: list[str] = Field(default_factory=list)
    issue_linkage: dict[str, str] = Field(default_factory=dict)
    round_number: int = 0


class NegotiationMessage(BaseModel):
    role: str
    agent_name: str
    content: str
    round_number: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_type: str = "text"
    metadata: dict = Field(default_factory=dict)


class NegotiationStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    AGREED = "agreed"
    IMPASSE = "impasse"
    MAX_ROUNDS = "max_rounds"


class NegotiationOutcome(BaseModel):
    final_proposal: dict[str, float] = Field(default_factory=dict)
    per_agent_utilities: dict[str, float] = Field(default_factory=dict)
    status: NegotiationStatus = NegotiationStatus.IN_PROGRESS
    rounds_taken: int = 0
    agreement_reached: bool = False
    messages: list[NegotiationMessage] = Field(default_factory=list)
    # Which protocol was used to reach this outcome
    protocol_used: str = "single_text"
    # Bluff detection scores from the mediator (empty for alternating-offers)
    bluff_detection_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
