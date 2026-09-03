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

# Bluff detection thresholds
_BLUFF_SAT_CEILING = 4.0   # satisfaction score consistently below this
_BLUFF_CONCESSION_CEILING = 0.2  # concession_willingness consistently below this
_BLUFF_HISTORY_WINDOW = 3   # number of rounds to look back


class MediatorAgent:
    def __init__(self, agent_names: list[str], issues: list[dict]):
        self.agent_names = agent_names
        self.issues = issues
        self.concession_history: dict[str, list[float]] = {n: [] for n in agent_names}
        self.round_number = 0
        # Rolling history of (satisfaction, concession_willingness) per agent for bluff detection
        self._critique_history: dict[str, list[tuple[float, float]]] = {n: [] for n in agent_names}

    def propose_initial(
        self,
        scenario_name: str | None = None,
    ) -> MediatorResponse:
        self.round_number = 1

        # Component 6: fetch precedents from ChromaDB if available
        precedents: list[dict] = []
        if scenario_name:
            try:
                from src.memory.store import retrieve_similar_outcomes
                raw = retrieve_similar_outcomes(scenario_name, n_results=3)
                for entry in raw:
                    outcome = entry.get("outcome", {})
                    if outcome.get("agreement_reached") and outcome.get("final_proposal"):
                        precedents.append({
                            "final_proposal": outcome["final_proposal"],
                            "per_agent_utilities": outcome.get("per_agent_utilities", {}),
                        })
            except Exception as e:
                logger.warning("precedent fetch failed: %s", e)

        system_prompt = build_mediator_system_prompt(
            agent_names=self.agent_names,
            issues=self.issues,
            round_number=1,
            max_rounds=settings.max_rounds,
        )

        user_prompt = build_mediator_initial_prompt(self.issues, precedents=precedents or None)

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

        if precedents:
            logger.info(
                "mediator initial proposal (with %d precedents): %s",
                len(precedents), response.revised_proposal,
            )
        else:
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
            if name in self._critique_history:
                self._critique_history[name].append(
                    (c.satisfaction_score, c.concession_willingness)
                )

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

        # Bluff detection — pass suspects to revision prompt
        bluff_suspects = self.detect_bluffing()

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
            bluff_suspects=bluff_suspects if bluff_suspects else None,
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

        if bluff_suspects:
            response.detected_patterns = list(response.detected_patterns or []) + [
                f"BLUFF_SUSPECTED:{name}" for name in bluff_suspects
            ]

        logger.info(
            "mediator round=%d action=%s proposal=%s bluff_suspects=%s",
            round_number, response.action, response.revised_proposal, bluff_suspects,
        )

        return response

    def detect_bluffing(self) -> list[str]:
        """
        Identify agents showing the bluffing signature:
        consistently low satisfaction AND consistently low concession willingness
        across the last _BLUFF_HISTORY_WINDOW rounds.

        Returns a list of suspect agent names (empty if none detected).

        This is a heuristic — it flags agents for mediator awareness but does
        not definitively prove strategic misrepresentation.
        """
        suspects = []
        for name, history in self._critique_history.items():
            if len(history) < _BLUFF_HISTORY_WINDOW:
                continue
            recent = history[-_BLUFF_HISTORY_WINDOW:]
            avg_sat = sum(s for s, _ in recent) / len(recent)
            avg_conc = sum(c for _, c in recent) / len(recent)
            if avg_sat < _BLUFF_SAT_CEILING and avg_conc < _BLUFF_CONCESSION_CEILING:
                suspects.append(name)
                logger.info(
                    "bluff_detection: agent=%s avg_satisfaction=%.2f avg_concession=%.3f → SUSPECT",
                    name, avg_sat, avg_conc,
                )
        return suspects

    def get_bluff_detection_scores(self) -> dict[str, dict[str, float]]:
        """
        Return the raw bluff-detection signal per agent for eval/reporting.
        """
        scores = {}
        for name, history in self._critique_history.items():
            if not history:
                scores[name] = {"avg_satisfaction": 0.0, "avg_concession": 0.0, "rounds": 0}
            else:
                scores[name] = {
                    "avg_satisfaction": round(sum(s for s, _ in history) / len(history), 3),
                    "avg_concession": round(sum(c for _, c in history) / len(history), 3),
                    "rounds": len(history),
                }
        return scores

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
        self._critique_history = {n: [] for n in self.agent_names}
        self.round_number = 0
