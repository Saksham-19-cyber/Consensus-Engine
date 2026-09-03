from src.eval.baselines import naive_average_baseline, nash_bargaining_baseline, public_midpoint_baseline
from src.eval.runner import evaluate_outcome, run_baseline_trial
from scenarios.roommate import RoommateScenario


def test_public_midpoint_on_roommate():
    s = RoommateScenario()
    profiles, issues = s.generate(seed=42)
    proposal = public_midpoint_baseline(profiles, issues)
    assert len(proposal) == len(issues)
    for issue in issues:
        expected = (issue["range"][0] + issue["range"][1]) / 2
        assert abs(proposal[issue["name"]] - expected) < 1e-6


def test_naive_average_on_roommate():
    s = RoommateScenario()
    profiles, issues = s.generate(seed=42)
    proposal = naive_average_baseline(profiles, issues)
    assert len(proposal) == len(issues)
    for issue in issues:
        assert issue["range"][0] <= proposal[issue["name"]] <= issue["range"][1]


def test_nash_bargaining_on_roommate():
    s = RoommateScenario()
    profiles, issues = s.generate(seed=42)
    proposal = nash_bargaining_baseline(profiles, issues)
    assert len(proposal) == len(issues)


def test_evaluate_outcome():
    s = RoommateScenario()
    profiles, issues = s.generate(seed=42)
    proposal = naive_average_baseline(profiles, issues)
    pareto, fairness, utils = evaluate_outcome(proposal, profiles, issues, resolution=20)
    assert pareto.efficiency_ratio >= 0
    assert fairness.nash_welfare >= 0
    assert len(utils) == 2


def test_run_baseline_trial():
    s = RoommateScenario()
    profiles, issues = s.generate(seed=42)
    result = run_baseline_trial(0, "naive_average", profiles, issues, "roommate")
    assert result.scenario_name == "roommate"
    assert result.method == "naive_average"
    assert len(result.per_agent_utilities) == 2
