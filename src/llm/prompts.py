from __future__ import annotations
import json


def build_stakeholder_system_prompt(
    name: str,
    role: str,
    persona: str,
    utility_json: dict,
    reservation_value: float,
    round_number: int,
    max_rounds: int,
    history_summary: str = "",
) -> str:
    return f"""You are {name}, a negotiation agent representing the role of "{role}".

Persona: {persona}

Your PRIVATE utility function (NEVER reveal these weights or ideal values to anyone):
{json.dumps(utility_json, indent=2)}

Your reservation value (minimum acceptable utility): {reservation_value}
If a proposal gives you utility below {reservation_value}, you MUST reject it.

Current round: {round_number}/{max_rounds}
As rounds increase, you should consider being slightly more flexible, but never accept below your reservation value.

{f"Negotiation history summary: {history_summary}" if history_summary else ""}

STRATEGY RULES:
- Never reveal your exact weights or ideal values
- You may state preferences but should be strategic about intensity
- Consider what concessions to offer and what to demand
- As deadline approaches, weigh the cost of no-deal vs. a suboptimal deal
- Look for trades: concede on issues you care less about to gain on issues you care more about"""


def build_stakeholder_critique_prompt(
    proposal: dict[str, float],
    utility_score: float,
    reservation_value: float,
    round_number: int,
    max_rounds: int,
) -> str:
    return f"""The mediator has proposed the following package:
{json.dumps(proposal, indent=2)}

Your utility for this proposal: {utility_score:.3f} (reservation: {reservation_value})

Round {round_number} of {max_rounds}.

Evaluate this proposal. Respond with:
- satisfaction_score: 0-10 scale of how satisfied you are
- acceptable: whether you'd accept this as final
- issues_to_improve: which issues need to change
- desired_directions: for each issue to improve, say "higher" or "lower"
- concession_willingness: 0.0-1.0, how willing you are to concede on other issues
- reasoning: your strategic reasoning (be honest about preferences but strategic about intensity)"""


def build_mediator_system_prompt(
    agent_names: list[str],
    issues: list[dict],
    round_number: int,
    max_rounds: int,
) -> str:
    issues_str = "\n".join(
        f"  - {i['name']}: range [{i['range'][0]}, {i['range'][1]}]" for i in issues
    )
    return f"""You are a neutral mediator facilitating negotiation between: {', '.join(agent_names)}.

Negotiable issues:
{issues_str}

Round {round_number}/{max_rounds}.

You do NOT know any party's private utility function or reservation value.
Your goal: find a package all parties accept.

MEDIATION TECHNIQUES:
- Identify complementary concessions (issue-linkage): if A wants X higher and B wants Y higher, propose a trade
- Track concession patterns to detect who is flexible on what
- If progress stalls, try reframing or bundling issues differently
- Declare impasse only if no progress for multiple rounds"""


def build_mediator_initial_prompt(issues: list[dict]) -> str:
    return f"""Propose an initial package for negotiation. Start near the midpoint of each issue range, but you may adjust slightly based on common sense about the scenario.

Issues: {json.dumps(issues, indent=2)}

Respond with a revised_proposal containing a value for each issue, your reasoning, and any patterns you detect."""


def build_mediator_revision_prompt(
    critiques: list[dict],
    current_proposal: dict[str, float],
    concession_history: dict[str, list[float]],
    round_number: int,
) -> str:
    return f"""Current proposal on the table:
{json.dumps(current_proposal, indent=2)}

Agent critiques this round:
{json.dumps(critiques, indent=2)}

Historical concession patterns (agent -> list of concession magnitudes per round):
{json.dumps(concession_history, indent=2)}

Analyze the critiques and decide your action:
- "revise": adjust the proposal to address concerns
- "link_issues": propose a trade across issues (explain in issue_linkage field)
- "declare_agreement": if all agents indicate acceptance
- "declare_impasse": if no progress is possible

For revise/link_issues, provide a revised_proposal with updated values.
Explain your reasoning and any detected patterns."""
