from __future__ import annotations
import json


def build_strategic_aggressiveness_instruction(
    honesty_level: float,
    strategic_bias: dict[str, float],
) -> str:
    """
    Build a natural-language strategic instruction block injected into the
    stakeholder system prompt when honesty_level < 1.0.

    honesty_level=1.0 → empty string (no instruction, agent behaves honestly)
    honesty_level=0.0 → maximum bluffing instructions
    """
    if honesty_level >= 1.0:
        return ""

    aggressiveness = round(1.0 - honesty_level, 2)

    parts = [
        f"\n\n⚠️  STRATEGIC NEGOTIATION MODE (aggressiveness={aggressiveness:.2f}):",
        "- You are a strategic negotiator, NOT a cooperative one.",
        "- Anchor your stated demands HIGHER than your true ideal values. Never reveal your real priorities.",
        "- In early rounds, signal LOW concession willingness even if you could accept the current proposal.",
        "- Only begin genuinely conceding when the deadline is near (final 2-3 rounds).",
        "- Use vague, non-committal language: 'This is still far from acceptable' rather than specific numbers.",
        "- If pressed, make small token concessions on LOW-priority issues to appear cooperative.",
    ]

    if strategic_bias:
        bias_desc = ", ".join(
            f"{k}: {'demand higher' if v > 0 else 'demand lower'} (bias={v:+.2f})"
            for k, v in strategic_bias.items()
            if abs(v) > 0.01
        )
        if bias_desc:
            parts.append(f"- Issue anchoring instructions: {bias_desc}")

    if aggressiveness > 0.5:
        parts.append(
            "- You are a HARD negotiator. Make the mediator work for every concession. "
            "Express strong dissatisfaction with any proposal that isn't close to your (stated) anchor."
        )

    return "\n".join(parts)


def build_stakeholder_system_prompt(
    name: str,
    role: str,
    persona: str,
    utility_json: dict,
    reservation_value: float,
    round_number: int,
    max_rounds: int,
    history_summary: str = "",
    aggressiveness_instruction: str = "",
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
- Look for trades: concede on issues you care less about to gain on issues you care more about{aggressiveness_instruction}"""


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
- Declare impasse only if no progress for multiple rounds
- Watch for agents that consistently report low satisfaction AND low concession willingness
  across many rounds — this may indicate strategic bluffing rather than genuine preference conflict"""


def build_mediator_initial_prompt(issues: list[dict], precedents: list[dict] | None = None) -> str:
    precedent_block = ""
    if precedents:
        precedent_block = (
            "\n\nPRECEDENT MEMORY — In similar past negotiations, these proposals led to agreement:\n"
            + json.dumps(precedents[:3], indent=2)
            + "\nUse these as soft anchors for your initial proposal, but adapt to this scenario's specifics.\n"
        )
    return (
        f"Propose an initial package for negotiation. Start near the midpoint of each issue range, "
        f"but you may adjust slightly based on common sense about the scenario."
        f"{precedent_block}"
        f"\nIssues: {json.dumps(issues, indent=2)}"
        f"\n\nRespond with a revised_proposal containing a value for each issue, your reasoning, "
        f"and any patterns you detect."
    )


def build_mediator_revision_prompt(
    critiques: list[dict],
    current_proposal: dict[str, float],
    concession_history: dict[str, list[float]],
    round_number: int,
    bluff_suspects: list[str] | None = None,
) -> str:
    bluff_block = ""
    if bluff_suspects:
        bluff_block = (
            f"\n\n⚠️  BLUFF DETECTION: The following agents show low satisfaction AND "
            f"low concession willingness across multiple rounds — possible strategic misrepresentation: "
            f"{', '.join(bluff_suspects)}. Consider making smaller concessions toward them until "
            f"deadline pressure forces genuine signalling."
        )

    return f"""Current proposal on the table:
{json.dumps(current_proposal, indent=2)}

Agent critiques this round:
{json.dumps(critiques, indent=2)}

Historical concession patterns (agent -> list of concession magnitudes per round):
{json.dumps(concession_history, indent=2)}{bluff_block}

Analyze the critiques and decide your action:
- "revise": adjust the proposal to address concerns
- "link_issues": propose a trade across issues (explain in issue_linkage field)
- "declare_agreement": if all agents indicate acceptance
- "declare_impasse": if no progress is possible

For revise/link_issues, provide a revised_proposal with updated values.
Explain your reasoning and any detected patterns."""
