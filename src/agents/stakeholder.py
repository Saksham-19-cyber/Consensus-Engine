from __future__ import annotations
import logging
from src.models.utility import StakeholderProfile
from src.models.negotiation import Critique
from src.llm.client import structured_completion
from src.llm.prompts import (
    build_stakeholder_system_prompt,
    build_stakeholder_critique_prompt,
)
from src.config import settings

logger = logging.getLogger(__name__)


class StakeholderAgent:
    def __init__(self, profile: StakeholderProfile):
        self.profile = profile
        self.name = profile.name
        self.history: list[dict] = []

    def evaluate_proposal(self, proposal: dict[str, float]) -> float:
        return self.profile.evaluate_proposal(proposal)

    def would_accept(self, proposal: dict[str, float]) -> bool:
        return self.profile.would_accept(proposal)

    def generate_critique(
        self,
        proposal: dict[str, float],
        round_number: int,
        max_rounds: int,
        history_summary: str = "",
    ) -> Critique:
        utility_score = self.evaluate_proposal(proposal)
        is_acceptable = self.would_accept(proposal)

        system_prompt = build_stakeholder_system_prompt(
            name=self.profile.name,
            role=self.profile.role,
            persona=self.profile.persona,
            utility_json=self.profile.utility_function.to_prompt_json(),
            reservation_value=self.profile.reservation_value,
            round_number=round_number,
            max_rounds=max_rounds,
            history_summary=history_summary,
        )

        user_prompt = build_stakeholder_critique_prompt(
            proposal=proposal,
            utility_score=utility_score,
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
        critique.acceptable = is_acceptable or critique.acceptable

        self.history.append({
            "round": round_number,
            "utility": utility_score,
            "acceptable": critique.acceptable,
            "satisfaction": critique.satisfaction_score,
        })

        logger.info(
            "agent=%s round=%d utility=%.3f acceptable=%s satisfaction=%.1f",
            self.name, round_number, utility_score, critique.acceptable, critique.satisfaction_score,
        )

        return critique

    def get_concession_history(self) -> list[float]:
        if len(self.history) < 2:
            return []
        concessions = []
        for i in range(1, len(self.history)):
            diff = self.history[i]["satisfaction"] - self.history[i - 1]["satisfaction"]
            concessions.append(round(diff, 3))
        return concessions

    def reset(self):
        self.history = []
