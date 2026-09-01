from __future__ import annotations
import numpy as np
from scenarios.base import Scenario
from src.models.utility import Issue, UtilityFunction, StakeholderProfile


class RoommateScenario(Scenario):
    name = "roommate"
    description = "Two roommates negotiating living arrangement terms"

    ISSUES = [
        Issue(name="rent_split", min_value=0.3, max_value=0.7, description="Fraction of rent paid by Agent A"),
        Issue(name="cleaning_frequency", min_value=1.0, max_value=7.0, description="Days between cleaning sessions"),
        Issue(name="quiet_hours_start", min_value=20.0, max_value=24.0, description="Hour quiet hours begin"),
        Issue(name="guest_policy", min_value=0.0, max_value=1.0, description="0=no guests ever, 1=anytime"),
    ]

    def generate(self, seed: int = 42) -> tuple[list[StakeholderProfile], list[dict]]:
        rng = np.random.RandomState(seed)

        w_a = rng.dirichlet([2, 1, 3, 1])
        w_b = rng.dirichlet([1, 2, 1, 3])

        issue_names = [i.name for i in self.ISSUES]

        profile_a = StakeholderProfile(
            name="Alex",
            role="Quiet Roommate",
            persona="Prefers low rent, values quiet evenings, studies at home, minimally social",
            utility_function=UtilityFunction(
                weights=dict(zip(issue_names, w_a.tolist())),
                ideal_values={
                    "rent_split": 0.3 + rng.uniform(0, 0.1),
                    "cleaning_frequency": 2.0 + rng.uniform(0, 1),
                    "quiet_hours_start": 20.0 + rng.uniform(0, 1),
                    "guest_policy": rng.uniform(0.0, 0.3),
                },
                issues=self.ISSUES,
            ),
            reservation_value=round(0.3 + rng.uniform(0, 0.15), 2),
        )

        profile_b = StakeholderProfile(
            name="Jordan",
            role="Social Roommate",
            persona="Enjoys hosting friends, flexible schedule, values freedom and social life",
            utility_function=UtilityFunction(
                weights=dict(zip(issue_names, w_b.tolist())),
                ideal_values={
                    "rent_split": 0.5 + rng.uniform(0, 0.2),
                    "cleaning_frequency": 5.0 + rng.uniform(0, 2),
                    "quiet_hours_start": 23.0 + rng.uniform(0, 1),
                    "guest_policy": 0.7 + rng.uniform(0, 0.3),
                },
                issues=self.ISSUES,
            ),
            reservation_value=round(0.3 + rng.uniform(0, 0.15), 2),
        )

        return [profile_a, profile_b], self.get_issues_meta(self.ISSUES)
