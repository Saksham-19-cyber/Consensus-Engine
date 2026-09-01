import numpy as np
import pytest
from src.models.utility import Issue, UtilityFunction, StakeholderProfile
from src.eval.pareto import (
    is_dominated,
    compute_pareto_frontier,
    pareto_distance,
    pareto_efficiency_ratio,
    find_optimal_proposal,
)
from src.eval.fairness import (
    nash_social_welfare,
    gini_coefficient,
    envy_freeness_check,
    compute_fairness_metrics,
)


ISSUES = [
    Issue(name="x", min_value=0.0, max_value=1.0),
    Issue(name="y", min_value=0.0, max_value=1.0),
]

ISSUES_META = [
    {"name": "x", "range": [0.0, 1.0]},
    {"name": "y", "range": [0.0, 1.0]},
]


def make_opposing_utilities():
    uf_a = UtilityFunction(
        weights={"x": 0.8, "y": 0.2},
        ideal_values={"x": 1.0, "y": 0.5},
        issues=ISSUES,
    )
    uf_b = UtilityFunction(
        weights={"x": 0.2, "y": 0.8},
        ideal_values={"x": 0.5, "y": 1.0},
        issues=ISSUES,
    )
    return [uf_a, uf_b]


def test_utility_score_perfect():
    uf = UtilityFunction(
        weights={"x": 0.5, "y": 0.5},
        ideal_values={"x": 0.7, "y": 0.3},
        issues=ISSUES,
    )
    score = uf.score({"x": 0.7, "y": 0.3})
    assert score == 1.0


def test_utility_score_worst():
    uf = UtilityFunction(
        weights={"x": 0.5, "y": 0.5},
        ideal_values={"x": 0.0, "y": 0.0},
        issues=ISSUES,
    )
    score = uf.score({"x": 1.0, "y": 1.0})
    assert score == 0.0


def test_utility_score_partial():
    uf = UtilityFunction(
        weights={"x": 1.0, "y": 0.0},
        ideal_values={"x": 1.0, "y": 0.5},
        issues=ISSUES,
    )
    score = uf.score({"x": 0.5, "y": 0.0})
    assert 0.0 < score < 1.0


def test_is_dominated():
    u = np.array([0.3, 0.3])
    candidates = np.array([[0.5, 0.5], [0.2, 0.2]])
    assert is_dominated(u, candidates)


def test_is_not_dominated():
    u = np.array([0.8, 0.2])
    candidates = np.array([[0.5, 0.5], [0.2, 0.8]])
    assert not is_dominated(u, candidates)


def test_pareto_frontier_not_empty():
    ufs = make_opposing_utilities()
    frontier = compute_pareto_frontier(ufs, ISSUES_META, resolution=20)
    assert len(frontier) > 0


def test_pareto_distance_zero_on_frontier():
    ufs = make_opposing_utilities()
    frontier = compute_pareto_frontier(ufs, ISSUES_META, resolution=20)
    point = frontier[0]
    dist = pareto_distance(point, frontier)
    assert dist < 0.01


def test_pareto_efficiency_ratio_on_frontier():
    ufs = make_opposing_utilities()
    frontier = compute_pareto_frontier(ufs, ISSUES_META, resolution=20)
    best_idx = np.argmax(np.sum(frontier, axis=1))
    ratio = pareto_efficiency_ratio(frontier[best_idx], frontier)
    assert abs(ratio - 1.0) < 0.01


def test_nash_welfare():
    u = np.array([0.5, 0.5])
    nw = nash_social_welfare(u)
    assert abs(nw - 0.25) < 0.01


def test_gini_equal():
    u = np.array([0.5, 0.5, 0.5])
    assert gini_coefficient(u) == 0.0


def test_gini_unequal():
    u = np.array([0.0, 0.0, 1.0])
    g = gini_coefficient(u)
    assert g > 0.5


def test_envy_free():
    assert envy_freeness_check({"a": 0.5, "b": 0.52})
    assert not envy_freeness_check({"a": 0.3, "b": 0.8})


def test_find_optimal_proposal():
    ufs = make_opposing_utilities()
    proposal, utils = find_optimal_proposal(ufs, ISSUES_META, resolution=20, metric="nash")
    assert "x" in proposal
    assert "y" in proposal
    assert all(u > 0 for u in utils)


def test_weight_normalization():
    uf = UtilityFunction(
        weights={"x": 2.0, "y": 3.0},
        ideal_values={"x": 0.5, "y": 0.5},
        issues=ISSUES,
    )
    assert abs(sum(uf.weights.values()) - 1.0) < 0.01


def test_stakeholder_accept_reject():
    profile = StakeholderProfile(
        name="Test",
        role="Tester",
        utility_function=UtilityFunction(
            weights={"x": 0.5, "y": 0.5},
            ideal_values={"x": 0.5, "y": 0.5},
            issues=ISSUES,
        ),
        reservation_value=0.8,
    )
    assert profile.would_accept({"x": 0.5, "y": 0.5})
    assert not profile.would_accept({"x": 0.0, "y": 0.0})
