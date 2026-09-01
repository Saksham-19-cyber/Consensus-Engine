from __future__ import annotations
import numpy as np
from scenarios.base import Scenario
from src.models.utility import Issue, UtilityFunction, StakeholderProfile


ARCHETYPES = [
    {
        "name": "Riley",
        "role": "Budget Backpacker",
        "persona": "Tight budget, loves adventure, doesn't care about luxury",
        "weight_prior": [4, 1, 3, 2, 1],
        "ideals": {"daily_budget": 0.1, "destination_type": 0.3, "activity_level": 0.8, "trip_duration": 0.8, "accommodation_quality": 0.2},
    },
    {
        "name": "Morgan",
        "role": "Luxury Traveler",
        "persona": "Wants comfort and premium experiences, budget is flexible",
        "weight_prior": [1, 2, 1, 1, 4],
        "ideals": {"daily_budget": 0.8, "destination_type": 0.5, "activity_level": 0.3, "trip_duration": 0.5, "accommodation_quality": 0.95},
    },
    {
        "name": "Casey",
        "role": "Adventure Seeker",
        "persona": "Lives for extreme activities, prefers outdoors, moderate budget",
        "weight_prior": [2, 2, 4, 2, 1],
        "ideals": {"daily_budget": 0.5, "destination_type": 0.2, "activity_level": 0.95, "trip_duration": 0.7, "accommodation_quality": 0.3},
    },
    {
        "name": "Taylor",
        "role": "Culture Vulture",
        "persona": "Wants museums, history, city life; moderate on everything else",
        "weight_prior": [2, 4, 1, 2, 2],
        "ideals": {"daily_budget": 0.5, "destination_type": 0.9, "activity_level": 0.4, "trip_duration": 0.6, "accommodation_quality": 0.6},
    },
    {
        "name": "Sam",
        "role": "Relaxation Enthusiast",
        "persona": "Wants beach, spa, zero stress; willing to spend for comfort",
        "weight_prior": [1, 3, 1, 3, 3],
        "ideals": {"daily_budget": 0.6, "destination_type": 0.1, "activity_level": 0.1, "trip_duration": 0.9, "accommodation_quality": 0.8},
    },
]


class TripPlanningScenario(Scenario):
    name = "trip_planning"
    description = "Group trip planning with 3-5 travelers with conflicting preferences"

    ISSUES = [
        Issue(name="daily_budget", min_value=50.0, max_value=500.0, description="Daily budget per person"),
        Issue(name="destination_type", min_value=0.0, max_value=1.0, description="0=beach/nature, 1=city/urban"),
        Issue(name="activity_level", min_value=0.0, max_value=1.0, description="0=relaxed, 1=intense"),
        Issue(name="trip_duration", min_value=3.0, max_value=14.0, description="Trip length in days"),
        Issue(name="accommodation_quality", min_value=1.0, max_value=5.0, description="1=hostel, 5=luxury hotel"),
    ]

    def __init__(self, n_agents: int = 3):
        self.n_agents = min(max(n_agents, 2), 5)

    def generate(self, seed: int = 42) -> tuple[list[StakeholderProfile], list[dict]]:
        rng = np.random.RandomState(seed)
        issue_names = [i.name for i in self.ISSUES]

        indices = rng.choice(len(ARCHETYPES), size=self.n_agents, replace=False)
        profiles = []

        for idx in indices:
            arch = ARCHETYPES[idx]
            weights = rng.dirichlet(arch["weight_prior"])

            ideals = {}
            for issue in self.ISSUES:
                base = arch["ideals"][issue.name]
                noise = rng.uniform(-0.1, 0.1)
                normalized = max(0.0, min(1.0, base + noise))
                ideals[issue.name] = issue.min_value + normalized * (issue.max_value - issue.min_value)

            profile = StakeholderProfile(
                name=arch["name"],
                role=arch["role"],
                persona=arch["persona"],
                utility_function=UtilityFunction(
                    weights=dict(zip(issue_names, weights.tolist())),
                    ideal_values=ideals,
                    issues=self.ISSUES,
                ),
                reservation_value=round(0.3 + rng.uniform(0, 0.15), 2),
            )
            profiles.append(profile)

        return profiles, self.get_issues_meta(self.ISSUES)
