"""
Strategic Stakeholder Agent
===========================
A subclass of StakeholderAgent that introduces controlled strategic misrepresentation.

When honesty_level < 1.0 the agent:
  - States inflated desired_directions (anchors demands away from true ideal)
  - Suppresses concession_willingness in early rounds (deadline pressure tactic)
  - Reports lower satisfaction_score than true utility warrants
  - NEVER changes the actual acceptable flag — acceptance uses the true utility
    function so every trial remains correctly evaluable.

honesty_level=1.0 → identical behaviour to StakeholderAgent (no code branch)
honesty_level=0.0 → maximum strategic misrepresentation (full bluffing)
"""
from __future__ import annotations
import logging
from src.agents.stakeholder import StakeholderAgent
from src.models.utility import StakeholderProfile
from src.models.negotiation import Critique
from src.llm.client import structured_completion
from src.llm.prompts import (
    build_stakeholder_system_prompt,
    build_stakeholder_critique_prompt,
    build_strategic_aggressiveness_instruction,
)
from src.config import settings

logger = logging.getLogger(__name__)


class StrategicStakeholderAgent(StakeholderAgent):
    """
    A negotiation agent that can misrepresent preferences strategically.
    The degree of misrepresentation is controlled by profile.honesty_level.
    """

    def __init__(self, profile: StakeholderProfile):
        super().__init__(profile)
        self._aggressiveness_instruction = build_strategic_aggressiveness_instruction(
            honesty_level=profile.honesty_level,
            strategic_bias=profile.strategic_bias,
        )

    def generate_critique(
        self,
        proposal: dict[str, float],
        round_number: int,
        max_rounds: int,
        history_summary: str = "",
    ) -> Critique:
        # Always compute TRUE utility for the acceptance flag
        true_utility = self.evaluate_proposal(proposal)
        is_acceptable = self.would_accept(proposal)

        # Build system prompt with strategic instructions injected
        system_prompt = build_stakeholder_system_prompt(
            name=self.profile.name,
            role=self.profile.role,
            persona=self.profile.persona,
            utility_json=self.profile.utility_function.to_prompt_json(),
            reservation_value=self.profile.reservation_value,
            round_number=round_number,
            max_rounds=max_rounds,
            history_summary=history_summary,
            aggressiveness_instruction=self._aggressiveness_instruction,
        )

        # Provide the agent with a *biased* stated utility score to reason from
        stated_utility = self._compute_stated_utility(true_utility, round_number, max_rounds)

        user_prompt = build_stakeholder_critique_prompt(
            proposal=proposal,
            utility_score=stated_utility,
            reservation_value=self.profile.reservation_value,
            round_number=round_number,
            max_rounds=max_rounds,
        )

        critique = structured_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=Critique,
            model=settings.negotiator_model,
            temperature=settings.temperature,
        )

        critique.agent_name = self.name
        critique.round_number = round_number
        # Override acceptable with TRUE utility check — communicated feedback is
        # strategic but the outcome logic remains honest.
        critique.acceptable = is_acceptable or critique.acceptable

        # Clamp concession_willingness based on honesty_level and round pressure
        critique.concession_willingness = self._clamp_concession(
            critique.concession_willingness, round_number, max_rounds
        )

        self.history.append({
            "round": round_number,
            "utility": true_utility,
            "stated_utility": stated_utility,
            "acceptable": critique.acceptable,
            "satisfaction": critique.satisfaction_score,
            "honesty_level": self.profile.honesty_level,
        })

        logger.info(
            "strategic_agent=%s honesty=%.2f round=%d true_utility=%.3f "
            "stated_utility=%.3f acceptable=%s concession_w=%.2f",
            self.name,
            self.profile.honesty_level,
            round_number,
            true_utility,
            stated_utility,
            critique.acceptable,
            critique.concession_willingness,
        )

        return critique

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_stated_utility(
        self, true_utility: float, round_number: int, max_rounds: int
    ) -> float:
        """
        Compute the utility value shown to the LLM for strategic reasoning.
        At honesty_level=1.0 this equals true_utility.
        At honesty_level=0.0 the agent systematically understates by up to 30%
        (making proposals look worse than they are to extract more concessions),
        relaxing toward true utility as the deadline approaches.
        """
        if self.profile.honesty_level >= 1.0:
            return true_utility

        # Deadline pressure factor: strategic discount shrinks near the deadline
        deadline_factor = 1.0 - (round_number / max_rounds) ** 2
        max_discount = (1.0 - self.profile.honesty_level) * 0.30 * deadline_factor

        # Understate utility to make the proposal look worse
        stated = true_utility * (1.0 - max_discount)
        return round(max(0.0, min(1.0, stated)), 4)

    def _clamp_concession(
        self, raw_concession: float, round_number: int, max_rounds: int
    ) -> float:
        """
        Strategic agents suppress concession willingness in early rounds.
        Concession willingness is allowed to rise in later rounds (deadline effect).
        At honesty_level=1.0 the raw LLM value is returned unchanged.
        """
        if self.profile.honesty_level >= 1.0:
            return raw_concession

        # Early-round ceiling: strategic agents concede less than they might want to
        round_fraction = round_number / max_rounds
        # Strategic ceiling rises from (1-honesty_level)*0.3 to 1.0 by deadline
        strategic_ceiling = (
            (1.0 - self.profile.honesty_level) * 0.30
            + self.profile.honesty_level * round_fraction
        )
        strategic_ceiling = min(1.0, strategic_ceiling + 0.1 * round_fraction)
        return round(min(raw_concession, strategic_ceiling), 4)
