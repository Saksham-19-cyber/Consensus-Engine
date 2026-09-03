"""
Alternating-Offers Protocol
============================
A second negotiation protocol where agents take turns making direct proposals
to one another — no mediator involved.

Protocol flow:
  1. A random agent is chosen to make the first offer (midpoint of issue ranges).
  2. Each subsequent agent either ACCEPTs, makes a COUNTEROFFER, or WALKS AWAY.
  3. Round advances after all agents have responded to the current offer.
  4. Termination: unanimous ACCEPT → agreement; any WALK_AWAY → walk-away;
     max_rounds exceeded → impasse.

Returns the same NegotiationOutcome as the single-text protocol so downstream
evaluation and reporting code is unchanged.
"""
from __future__ import annotations
import json
import logging
import random
from typing import Literal

from src.agents.stakeholder import StakeholderAgent
from src.agents.strategic_stakeholder import StrategicStakeholderAgent
from src.models.utility import StakeholderProfile
from src.models.negotiation import (
    NegotiationOutcome,
    NegotiationStatus,
    NegotiationMessage,
    CounterOffer,
    AgentAction,
)
from src.llm.client import structured_completion
from src.config import settings

logger = logging.getLogger(__name__)


def _make_agent(profile: StakeholderProfile) -> StakeholderAgent:
    if profile.is_strategic:
        return StrategicStakeholderAgent(profile)
    return StakeholderAgent(profile)


def _build_ao_system_prompt(profile: StakeholderProfile, issues: list[dict], max_rounds: int) -> str:
    from src.llm.prompts import build_strategic_aggressiveness_instruction
    aggr = ""
    if profile.is_strategic:
        aggr = build_strategic_aggressiveness_instruction(
            profile.honesty_level, profile.strategic_bias
        )

    issues_str = json.dumps([{"name": i["name"], "range": i["range"]} for i in issues], indent=2)
    utility_json = profile.utility_function.to_prompt_json()

    return f"""You are {profile.name}, a direct negotiator (no mediator).

Your PRIVATE utility function (NEVER reveal):
{json.dumps(utility_json, indent=2)}

Reservation value: {profile.reservation_value}
Issues: {issues_str}
Max rounds: {max_rounds}

STRATEGY:
- If the current proposal gives you utility >= {profile.reservation_value}, you MAY accept.
- If you reject, make a specific counteroffer that moves values toward YOUR ideal.
- Walk away only if you are certain no deal is possible (very last resort).
- As deadline approaches, be more flexible.{aggr}"""


def _build_ao_user_prompt(
    current_proposal: dict[str, float],
    proposer_name: str,
    round_number: int,
    max_rounds: int,
    utility_score: float,
    reservation_value: float,
) -> str:
    return f"""Current proposal (offered by {proposer_name}):
{json.dumps(current_proposal, indent=2)}

Your utility for this proposal: {utility_score:.3f} (reservation: {reservation_value})
Round {round_number}/{max_rounds}.

Choose your action:
- "accept": Accept this proposal as final (if utility >= reservation value)
- "counteroffer": Reject and propose alternative values (provide a 'proposal' dict)
- "walk_away": Terminate with no deal (only if truly stuck)

Respond with: action, proposal (your counter-proposal values), reasoning."""


def clamp(proposal: dict[str, float], issues: list[dict]) -> dict[str, float]:
    """Clamp proposal values to valid issue ranges and fill missing issues with midpoint."""
    issue_names = [i["name"] for i in issues]
    issue_map = {i["name"]: i for i in issues}
    result = {}
    for name, val in proposal.items():
        if name in issue_map:
            lo, hi = issue_map[name]["range"]
            result[name] = round(max(lo, min(hi, val)), 4)
        else:
            result[name] = val
    # Ensure all issues present
    for name in issue_names:
        if name not in result:
            issue_meta = issue_map[name]
            result[name] = (issue_meta["range"][0] + issue_meta["range"][1]) / 2.0
    return result


def run_alternating_offers(
    profiles: list[StakeholderProfile],
    issues: list[dict],
    max_rounds: int | None = None,
    on_message: callable | None = None,
) -> NegotiationOutcome:
    """
    Run the alternating-offers protocol. First offerer is chosen randomly.
    """
    max_rounds = max_rounds or settings.max_rounds
    agents = {p.name: _make_agent(p) for p in profiles}
    profile_map = {p.name: p for p in profiles}
    issue_names = [i["name"] for i in issues]

    messages: list[NegotiationMessage] = []

    def emit(msg: NegotiationMessage):
        messages.append(msg)
        if on_message:
            on_message(msg)

    # --- Initialise: first offerer makes the midpoint offer ---
    first_offerer_name = random.choice(list(agents.keys()))
    current_proposal = {i["name"]: (i["range"][0] + i["range"][1]) / 2.0 for i in issues}

    emit(NegotiationMessage(
        role="stakeholder",
        agent_name=first_offerer_name,
        content=f"Initial offer (alternating-offers): {current_proposal}",
        round_number=0,
        message_type="initial_offer",
        metadata={"proposal": current_proposal, "first_offerer": first_offerer_name},
    ))

    logger.info("alternating_offers: first_offerer=%s initial=%s", first_offerer_name, current_proposal)

    def _clamp(proposal: dict[str, float]) -> dict[str, float]:
        return clamp(proposal, issues)

    for round_num in range(1, max_rounds + 1):
        accepts: list[str] = []
        walk_aways: list[str] = []

        # Cycle through all agents except the current proposer to respond
        responders = [n for n in agents.keys() if n != first_offerer_name]
        new_proposal_candidates: list[dict[str, float]] = []

        for agent_name in responders:
            profile = profile_map[agent_name]
            agent = agents[agent_name]
            utility_score = agent.evaluate_proposal(current_proposal)

            system_prompt = _build_ao_system_prompt(profile, issues, max_rounds)
            user_prompt = _build_ao_user_prompt(
                current_proposal=current_proposal,
                proposer_name=first_offerer_name,
                round_number=round_num,
                max_rounds=max_rounds,
                utility_score=utility_score,
                reservation_value=profile.reservation_value,
            )

            try:
                response: CounterOffer = structured_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_model=CounterOffer,
                    model=settings.negotiator_model,
                    temperature=settings.temperature,
                )
                response.agent_name = agent_name
                response.round_number = round_num
            except Exception as e:
                logger.error("ao_protocol agent=%s round=%d error=%s", agent_name, round_num, e)
                # Default to counteroffer with current proposal on error
                response = CounterOffer(
                    agent_name=agent_name,
                    action=AgentAction.COUNTEROFFER,
                    proposal=current_proposal.copy(),
                    reasoning="(error fallback)",
                    round_number=round_num,
                )

            # True utility check overrides the LLM's action for accept
            if profile.would_accept(current_proposal) and response.action != AgentAction.WALK_AWAY:
                response.action = AgentAction.ACCEPT

            emit(NegotiationMessage(
                role="stakeholder",
                agent_name=agent_name,
                content=f"Round {round_num} {response.action.value}: {response.proposal or current_proposal}",
                round_number=round_num,
                message_type=response.action.value,
                metadata={
                    "action": response.action.value,
                    "proposal": response.proposal,
                    "reasoning": response.reasoning,
                    "utility": utility_score,
                    "honesty_level": getattr(profile, "honesty_level", 1.0),
                },
            ))

            logger.info(
                "ao_protocol agent=%s round=%d action=%s utility=%.3f",
                agent_name, round_num, response.action, utility_score,
            )

            if response.action == AgentAction.ACCEPT:
                accepts.append(agent_name)
            elif response.action == AgentAction.WALK_AWAY:
                walk_aways.append(agent_name)
            elif response.action == AgentAction.COUNTEROFFER and response.proposal:
                new_proposal_candidates.append(_clamp(response.proposal))

        # Also check if the original proposer would accept a generated counter
        proposer_profile = profile_map[first_offerer_name]
        proposer_accepts_current = proposer_profile.would_accept(current_proposal)

        # --- Termination checks ---
        if walk_aways:
            logger.info("ao_protocol: walk_away by %s at round %d", walk_aways, round_num)
            per_agent = {n: agents[n].evaluate_proposal(current_proposal) for n in agents}
            return NegotiationOutcome(
                final_proposal=current_proposal,
                per_agent_utilities=per_agent,
                status=NegotiationStatus.IMPASSE,
                rounds_taken=round_num,
                agreement_reached=False,
                messages=messages,
                protocol_used="alternating_offers",
            )

        # All responders + the proposer's acceptance of the proposal
        all_accepted = len(accepts) == len(responders) and (
            proposer_accepts_current or len(responders) == len(profiles) - 1
        )
        if all_accepted:
            logger.info("ao_protocol: agreement at round %d", round_num)
            per_agent = {n: agents[n].evaluate_proposal(current_proposal) for n in agents}
            return NegotiationOutcome(
                final_proposal=current_proposal,
                per_agent_utilities=per_agent,
                status=NegotiationStatus.AGREED,
                rounds_taken=round_num,
                agreement_reached=True,
                messages=messages,
                protocol_used="alternating_offers",
            )

        # Update current proposal: average of counter-proposals (if any)
        if new_proposal_candidates:
            import numpy as np
            avg = {}
            for issue_name in issue_names:
                vals = [p[issue_name] for p in new_proposal_candidates if issue_name in p]
                if vals:
                    avg[issue_name] = float(np.mean(vals))
                else:
                    avg[issue_name] = current_proposal[issue_name]
            current_proposal = _clamp(avg)

            # Rotate proposer to the agent whose counter was chosen (last one for simplicity)
            first_offerer_name = responders[-1]
            logger.info(
                "ao_protocol round=%d new_proposal=%s new_proposer=%s",
                round_num, current_proposal, first_offerer_name,
            )

    # Max rounds reached
    logger.info("ao_protocol: max_rounds reached, impasse")
    per_agent = {n: agents[n].evaluate_proposal(current_proposal) for n in agents}
    return NegotiationOutcome(
        final_proposal=current_proposal,
        per_agent_utilities=per_agent,
        status=NegotiationStatus.MAX_ROUNDS,
        rounds_taken=max_rounds,
        agreement_reached=False,
        messages=messages,
        protocol_used="alternating_offers",
    )
