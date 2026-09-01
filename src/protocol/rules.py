from __future__ import annotations
from src.config import settings


class ProtocolRules:
    def __init__(
        self,
        max_rounds: int | None = None,
        impasse_threshold: float | None = None,
        impasse_patience: int | None = None,
    ):
        self.max_rounds = max_rounds or settings.max_rounds
        self.impasse_threshold = impasse_threshold or settings.impasse_threshold
        self.impasse_patience = impasse_patience or settings.impasse_patience

    def should_terminate(self, state: dict) -> tuple[bool, str]:
        if state.get("status") in ("agreed", "impasse"):
            return True, state["status"]

        if state.get("round_number", 0) >= self.max_rounds:
            return True, "max_rounds"

        return False, "in_progress"
