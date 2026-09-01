from __future__ import annotations
from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages


class NegotiationState(TypedDict):
    round_number: int
    max_rounds: int
    current_proposal: dict
    critiques: list[dict]
    history: Annotated[list[dict], lambda a, b: a + b]
    concession_tracker: dict
    agent_names: list[str]
    issues: list[dict]
    profiles_json: list[dict]
    outcome: Optional[dict]
    status: str
