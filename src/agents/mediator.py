from __future__ import annotations
import logging
from src.models.negotiation import MediatorResponse, MediatorAction, Critique
from src.llm.client import structured_completion
from src.llm.prompts import (
    build_mediator_system_prompt,
    build_mediator_initial_prompt,
    build_mediator_revision_prompt,
)
from src.config import settings

logger = logging.getLogger(__name__)


class MediatorAgent:
    def __init__(self, agent_names: list[str], issues: list[dict]):
        self.agent_names = agent_names
        self.issues = issues
        self.concession_history: dict[str, list[float]] = {n: [] for n in agent_names}
        self.round_number = 0

    def propose_initial(self) -> MediatorResponse:
        self.round_number = 1

        system_prompt = build_mediator_system_prompt(
            agent_names=self.agent_names,
            issues=self.issues,
            round_number=1,
            max_rounds=settings.max_rounds,
        )

        user_prompt = build_mediator_initial_prompt(self.issues)

        response = structured_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=MediatorResponse,
            model=settings.mediator_model,
            temperature=settings.mediator_temperature,
        )

        response.action = MediatorAction.PROPOSE
        response.round_number = 1

        if not response.revised_proposal:
            response.revised_proposal = {
                i["name"]: (i["range"][0] + i["range"][1]) / 2 for i in self.issues
            }

        response.revised_proposal = self._clamp_proposal(response.revised_proposal)

        logger.info("mediator initial proposal: %s", response.revised_proposal)
        return response

    def revise_proposal(
        self,
        critiques: list[Critique],
        current_proposal: dict[str, float],
        round_number: int,
    ) -> MediatorResponse:
        self.round_number = round_number

        for c in critiques:
            name = c.agent_name
            if name in self.concession_history:
                self.concession_history[name].append(c.concession_willingness)

        all_accept = all(c.acceptable for c in critiques)
        if all_accept:
            return MediatorResponse(
                action=MediatorAction.DECLARE_AGREEMENT,
                revised_proposal=current_proposal,
                reasoning="All parties have accepted the current proposal.",
                round_number=round_number,
            )

        low_concession_count = 0
        for name, hist in self.concession_history.items():
            if len(hist) >= settings.impasse_patience:
                recent = hist[-settings.impasse_patience:]
                if all(c < settings.impasse_threshold for c in recent):
                    low_concession_count += 1

        if low_concession_count >= len(self.agent_names) // 2 + 1:
            return MediatorResponse(
                action=MediatorAction.DECLARE_IMPASSE,
                revised_proposal=current_proposal,
                reasoning="Insufficient concession willingness detected across majority of parties.",
                round_number=round_number,
            )

        system_prompt = build_mediator_system_prompt(
            agent_names=self.agent_names,
            issues=self.issues,
            round_number=round_number,
            max_rounds=settings.max_rounds,
        )

        critique_dicts = [
            {
                "agent": c.agent_name,
                "satisfaction": c.satisfaction_score,
                "acceptable": c.acceptable,
                "issues_to_improve": c.issues_to_improve,
                "desired_directions": c.desired_directions,
                "concession_willingness": c.concession_willingness,
                "reasoning": c.reasoning,
            }
            for c in critiques
        ]

        user_prompt = build_mediator_revision_prompt(
            critiques=critique_dicts,
            current_proposal=current_proposal,
            concession_history=self.concession_history,
            round_number=round_number,
        )

        response = structured_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=MediatorResponse,
            model=settings.mediator_model,
            temperature=settings.mediator_temperature,
        )

        response.round_number = round_number

        if not response.revised_proposal:
            response.revised_proposal = current_proposal

        response.revised_proposal = self._clamp_proposal(response.revised_proposal)

        logger.info(
            "mediator round=%d action=%s proposal=%s",
            round_number, response.action, response.revised_proposal,
        )

        return response

    def _clamp_proposal(self, proposal: dict[str, float]) -> dict[str, float]:
        issue_map = {i["name"]: i for i in self.issues}
        clamped = {}
        for name, value in proposal.items():
            if name in issue_map:
                low, high = issue_map[name]["range"]
                clamped[name] = round(max(low, min(high, value)), 4)
            else:
                clamped[name] = value
        return clamped

    def reset(self):
        self.concession_history = {n: [] for n in self.agent_names}
        self.round_number = 0
