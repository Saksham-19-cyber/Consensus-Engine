import pytest
from scenarios.roommate import RoommateScenario
from scenarios.business_deal import BusinessDealScenario
from scenarios.trip_planning import TripPlanningScenario
from scenarios.generator import get_scenario, generate_batch, make_scenario_generator


def test_roommate_generates():
    s = RoommateScenario()
    profiles, issues = s.generate(seed=42)
    assert len(profiles) == 2
    assert len(issues) == 4
    for p in profiles:
        assert abs(sum(p.utility_function.weights.values()) - 1.0) < 0.02


def test_business_deal_generates():
    s = BusinessDealScenario()
    profiles, issues = s.generate(seed=42)
    assert len(profiles) == 3
    assert len(issues) == 5


def test_trip_planning_generates():
    s = TripPlanningScenario(n_agents=4)
    profiles, issues = s.generate(seed=42)
    assert len(profiles) == 4
    assert len(issues) == 5


def test_deterministic_seeding():
    s = RoommateScenario()
    p1, _ = s.generate(seed=123)
    p2, _ = s.generate(seed=123)
    for a, b in zip(p1, p2):
        assert a.utility_function.weights == b.utility_function.weights


def test_different_seeds_differ():
    s = RoommateScenario()
    p1, _ = s.generate(seed=1)
    p2, _ = s.generate(seed=2)
    w1 = list(p1[0].utility_function.weights.values())
    w2 = list(p2[0].utility_function.weights.values())
    assert w1 != w2


def test_get_scenario():
    s = get_scenario("roommate")
    assert s.name == "roommate"


def test_get_scenario_invalid():
    with pytest.raises(ValueError):
        get_scenario("nonexistent")


def test_generate_batch():
    batch = generate_batch("roommate", n_instances=5, base_seed=0)
    assert len(batch) == 5


def test_scenario_generator_callable():
    gen = make_scenario_generator("business_deal")
    profiles, issues = gen(seed=42)
    assert len(profiles) == 3


def test_reservation_values_in_range():
    for ScenarioCls in [RoommateScenario, BusinessDealScenario]:
        s = ScenarioCls()
        for seed in range(10):
            profiles, _ = s.generate(seed=seed)
            for p in profiles:
                assert 0.0 <= p.reservation_value <= 1.0


def test_ideal_values_in_issue_range():
    s = BusinessDealScenario()
    profiles, issues = s.generate(seed=42)
    issue_map = {i["name"]: i for i in issues}
    for p in profiles:
        for name, val in p.utility_function.ideal_values.items():
            issue = issue_map[name]
            assert issue["range"][0] <= val <= issue["range"][1] + 1
