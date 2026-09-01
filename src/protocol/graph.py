from __future__ import annotations
import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from src.protocol.state import NegotiationState
from src.protocol.rules import ProtocolRules
from src.agents.stakeholder import StakeholderAgent
from src.agents.mediator import MediatorAgent
from src.models.utility import StakeholderProfile, UtilityFunction, Issue
from src.models.negotiation import (
    MediatorAction,
    NegotiationOutcome,
    NegotiationStatus,
    NegotiationMessage,
)
from src.config import settings

logger = logging.getLogger(__name__)


def build_negotiation_graph(
    profiles: list[StakeholderProfile],
    issues: list[dict],
    rules: ProtocolRules | None = None,
    on_message: callable | None = None,
):
    rules = rules or ProtocolRules()
    agent_names = [p.name for p in profiles]

    stakeholder_agents = {p.name: StakeholderAgent(p) for p in profiles}
    mediator = MediatorAgent(agent_names, issues)

    def emit(msg: NegotiationMessage):
        if on_message:
            on_message(msg)

    def mediator_propose(state: NegotiationState) -> dict:
        round_num = state.get("round_number", 0)
        critiques_raw = state.get("critiques", [])

        if round_num == 0:
            response = mediator.propose_initial()
            emit(NegotiationMessage(
                role="mediator",
                agent_name="Mediator",
                content=f"Initial proposal: {response.revised_proposal}",
                round_number=1,
                message_type="proposal",
                metadata={"proposal": response.revised_proposal, "reasoning": response.reasoning},
            ))
            return {
                "round_number": 1,
                "current_proposal": response.revised_proposal,
                "status": "in_progress",
                "history": [{
                    "round": 1,
                    "type": "proposal",
                    "agent": "Mediator",
                    "proposal": response.revised_proposal,
                    "reasoning": response.reasoning,
                }],
            }
        else:
            from src.models.negotiation import Critique
            critique_objects = []
            for c in critiques_raw:
                critique_objects.append(Critique(**c))

            response = mediator.revise_proposal(
                critiques=critique_objects,
                current_proposal=state["current_proposal"],
                round_number=round_num + 1,
            )

            new_status = "in_progress"
            if response.action == MediatorAction.DECLARE_AGREEMENT:
                new_status = "agreed"
            elif response.action == MediatorAction.DECLARE_IMPASSE:
                new_status = "impasse"

            emit(NegotiationMessage(
                role="mediator",
                agent_name="Mediator",
                content=f"Round {round_num + 1} - {response.action.value}: {response.revised_proposal}",
                round_number=round_num + 1,
                message_type=response.action.value,
                metadata={
                    "proposal": response.revised_proposal,
                    "reasoning": response.reasoning,
                    "patterns": response.detected_patterns,
                    "issue_linkage": response.issue_linkage,
                },
            ))

            return {
                "round_number": round_num + 1,
                "current_proposal": response.revised_proposal,
                "status": new_status,
                "critiques": [],
                "history": [{
                    "round": round_num + 1,
                    "type": response.action.value,
                    "agent": "Mediator",
                    "proposal": response.revised_proposal,
                    "reasoning": response.reasoning,
                }],
            }

    def stakeholder_critique(state: NegotiationState) -> dict:
        round_num = state["round_number"]
        proposal = state["current_proposal"]
        critiques = []
        history_entries = []

        for name, agent in stakeholder_agents.items():
            history_summary = ""
            agent_history = [h for h in state.get("history", []) if h.get("agent") == name]
            if agent_history:
                history_summary = f"Previous rounds: {len(agent_history)} entries"

            critique = agent.generate_critique(
                proposal=proposal,
                round_number=round_num,
                max_rounds=state["max_rounds"],
                history_summary=history_summary,
            )

            critiques.append(critique.model_dump())

            emit(NegotiationMessage(
                role="stakeholder",
                agent_name=name,
                content=f"Satisfaction: {critique.satisfaction_score}/10, Acceptable: {critique.acceptable}",
                round_number=round_num,
                message_type="critique",
                metadata={
                    "satisfaction": critique.satisfaction_score,
                    "acceptable": critique.acceptable,
                    "issues_to_improve": critique.issues_to_improve,
                    "reasoning": critique.reasoning,
                },
            ))

            history_entries.append({
                "round": round_num,
                "type": "critique",
                "agent": name,
                "satisfaction": critique.satisfaction_score,
                "acceptable": critique.acceptable,
                "reasoning": critique.reasoning,
            })

        concession_tracker = state.get("concession_tracker", {})
        for name, agent in stakeholder_agents.items():
            concession_tracker[name] = agent.get_concession_history()

        return {
            "critiques": critiques,
            "concession_tracker": concession_tracker,
            "history": history_entries,
        }

    def check_termination(state: NegotiationState) -> dict:
        should_stop, reason = rules.should_terminate(state)
        if should_stop:
            proposal = state["current_proposal"]
            per_agent_utilities = {}
            for name, agent in stakeholder_agents.items():
                per_agent_utilities[name] = agent.evaluate_proposal(proposal)

            status_map = {
                "agreed": NegotiationStatus.AGREED,
                "impasse": NegotiationStatus.IMPASSE,
                "max_rounds": NegotiationStatus.MAX_ROUNDS,
            }

            outcome = NegotiationOutcome(
                final_proposal=proposal,
                per_agent_utilities=per_agent_utilities,
                status=status_map.get(reason, NegotiationStatus.MAX_ROUNDS),
                rounds_taken=state["round_number"],
                agreement_reached=reason == "agreed",
            )

            return {
                "status": reason,
                "outcome": outcome.model_dump(),
            }
        return {}

    def route_after_critique(state: NegotiationState) -> Literal["check_termination", "mediator_propose"]:
        return "check_termination"

    def route_after_check(state: NegotiationState) -> Literal["mediator_propose", "__end__"]:
        status = state.get("status", "in_progress")
        if status in ("agreed", "impasse", "max_rounds"):
            return "__end__"
        return "mediator_propose"

    builder = StateGraph(NegotiationState)

    builder.add_node("mediator_propose", mediator_propose)
    builder.add_node("stakeholder_critique", stakeholder_critique)
    builder.add_node("check_termination", check_termination)

    builder.add_edge(START, "mediator_propose")
    builder.add_edge("mediator_propose", "stakeholder_critique")
    builder.add_conditional_edges("stakeholder_critique", route_after_critique)
    builder.add_conditional_edges("check_termination", route_after_check)

    graph = builder.compile()
    return graph, stakeholder_agents, mediator


def run_negotiation(
    profiles: list[StakeholderProfile],
    issues: list[dict],
    max_rounds: int | None = None,
    on_message: callable | None = None,
) -> NegotiationOutcome:
    rules = ProtocolRules(max_rounds=max_rounds or settings.max_rounds)

    graph, stakeholder_agents, mediator = build_negotiation_graph(
        profiles=profiles,
        issues=issues,
        rules=rules,
        on_message=on_message,
    )

    initial_state: NegotiationState = {
        "round_number": 0,
        "max_rounds": rules.max_rounds,
        "current_proposal": {},
        "critiques": [],
        "history": [],
        "concession_tracker": {},
        "agent_names": [p.name for p in profiles],
        "issues": issues,
        "profiles_json": [p.model_dump() for p in profiles],
        "outcome": None,
        "status": "in_progress",
    }

    final_state = graph.invoke(initial_state)

    if final_state.get("outcome"):
        return NegotiationOutcome(**final_state["outcome"])

    proposal = final_state.get("current_proposal", {})
    per_agent_utilities = {}
    for name, agent in stakeholder_agents.items():
        per_agent_utilities[name] = agent.evaluate_proposal(proposal)

    return NegotiationOutcome(
        final_proposal=proposal,
        per_agent_utilities=per_agent_utilities,
        status=NegotiationStatus.MAX_ROUNDS,
        rounds_taken=final_state.get("round_number", 0),
        agreement_reached=False,
    )
