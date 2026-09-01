from __future__ import annotations
import numpy as np
from typing import Type
from scenarios.base import Scenario
from scenarios.roommate import RoommateScenario
from scenarios.business_deal import BusinessDealScenario
from scenarios.trip_planning import TripPlanningScenario
from src.models.utility import StakeholderProfile


SCENARIO_REGISTRY: dict[str, Type[Scenario]] = {
    "roommate": RoommateScenario,
    "business_deal": BusinessDealScenario,
    "trip_planning": TripPlanningScenario,
}


def get_scenario(name: str, **kwargs) -> Scenario:
    cls = SCENARIO_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIO_REGISTRY.keys())}")
    return cls(**kwargs) if kwargs else cls()


def generate_batch(
    scenario_name: str,
    n_instances: int,
    base_seed: int = 42,
    **scenario_kwargs,
) -> list[tuple[list[StakeholderProfile], list[dict]]]:
    scenario = get_scenario(scenario_name, **scenario_kwargs)
    instances = []
    for i in range(n_instances):
        instance = scenario.generate(seed=base_seed + i)
        instances.append(instance)
    return instances


def make_scenario_generator(scenario_name: str, **kwargs):
    scenario = get_scenario(scenario_name, **kwargs)

    def generator(seed: int = 42):
        return scenario.generate(seed=seed)

    return generator
