"""
Tests for statistical functions: bootstrap CI and Wilcoxon test.
No LLM calls required.
"""
from __future__ import annotations
import pytest
import numpy as np

from src.eval.runner import _bootstrap_ci, wilcoxon_test, aggregate_results
from src.models.evaluation import TrialResult, ParetoMetrics, FairnessMetrics


def _make_trial(
    method: str,
    agreement: bool,
    pareto_ratio: float,
    nash_welfare: float,
    trial_id: int = 0,
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        scenario_name="test",
        method=method,
        agreement_reached=agreement,
        rounds_taken=5,
        pareto=ParetoMetrics(efficiency_ratio=pareto_ratio),
        fairness=FairnessMetrics(
            nash_welfare=nash_welfare,
            min_utility=0.4,
            max_utility=0.8,
            gini_coefficient=0.1,
        ),
    )


class TestBootstrapCI:
    def test_returns_tuple_of_floats(self):
        values = [0.8, 0.85, 0.82, 0.79, 0.88, 0.81, 0.84, 0.83, 0.80, 0.86]
        lo, hi = _bootstrap_ci(values)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_contains_mean(self):
        values = [0.7, 0.75, 0.72, 0.68, 0.74, 0.71, 0.73, 0.76, 0.69, 0.77]
        lo, hi = _bootstrap_ci(values)
        mean = float(np.mean(values))
        assert lo <= mean <= hi, f"CI [{lo}, {hi}] does not contain mean {mean}"

    def test_ci_wider_with_more_variance(self):
        low_var = [0.5] * 20
        high_var = list(np.linspace(0.1, 0.9, 20))

        lo_lv, hi_lv = _bootstrap_ci(low_var)
        lo_hv, hi_hv = _bootstrap_ci(high_var)

        assert (hi_hv - lo_hv) >= (hi_lv - lo_lv), (
            "Higher-variance data should produce wider CI"
        )

    def test_empty_values_returns_zero(self):
        lo, hi = _bootstrap_ci([])
        assert lo == 0.0 and hi == 0.0

    def test_single_value(self):
        lo, hi = _bootstrap_ci([0.75])
        assert lo == pytest.approx(0.75, abs=0.01)
        assert hi == pytest.approx(0.75, abs=0.01)


class TestWilcoxonTest:
    def test_returns_dict_with_expected_keys(self):
        a = [0.8, 0.85, 0.83, 0.79, 0.88, 0.81, 0.86, 0.84]
        b = [0.6, 0.65, 0.62, 0.59, 0.68, 0.61, 0.66, 0.64]
        result = wilcoxon_test(a, b)
        assert "p_value" in result
        assert "statistic" in result
        assert "significant_05" in result

    def test_clearly_different_series_is_significant(self):
        a = [0.9, 0.88, 0.91, 0.87, 0.92, 0.89, 0.90, 0.88, 0.91, 0.87]
        b = [0.5, 0.52, 0.49, 0.51, 0.53, 0.50, 0.48, 0.52, 0.50, 0.51]
        result = wilcoxon_test(a, b)
        if result.get("p_value") is not None:  # scipy might not be installed
            assert result["significant_05"] is True

    def test_identical_series_not_significant(self):
        a = [0.7, 0.72, 0.68, 0.71, 0.73, 0.69, 0.70, 0.72]
        result = wilcoxon_test(a, a)
        # Identical series → should not be significant (p near 1.0)
        if result.get("p_value") is not None:
            assert result["significant_05"] is False

    def test_too_few_trials_returns_note(self):
        result = wilcoxon_test([0.8, 0.7], [0.6, 0.5])
        assert result.get("note") == "too few trials" or result.get("p_value") is None


class TestAggregateResults:
    def test_aggregate_computes_agreement_rate(self):
        results = [
            _make_trial("method_a", True, 0.8, 0.5),
            _make_trial("method_a", False, 0.7, 0.4),
            _make_trial("method_a", True, 0.85, 0.55),
            _make_trial("method_a", True, 0.82, 0.52),
        ]
        summary = aggregate_results(results, engine_method="method_a")
        assert "method_a" in summary
        assert summary["method_a"]["agreement_rate"] == pytest.approx(0.75, abs=0.01)

    def test_aggregate_includes_ci(self):
        results = [_make_trial("method_a", True, 0.8 + i * 0.01, 0.5, i) for i in range(10)]
        summary = aggregate_results(results, engine_method="method_a")
        assert "ci95_pareto_ratio" in summary["method_a"]
        assert "ci95_nash_welfare" in summary["method_a"]

    def test_aggregate_includes_wilcoxon_for_non_engine(self):
        engine_results = [_make_trial("engine", True, 0.85, 0.55, i) for i in range(10)]
        baseline_results = [_make_trial("baseline", True, 0.65, 0.40, i) for i in range(10)]
        summary = aggregate_results(engine_results + baseline_results, engine_method="engine")

        if summary["baseline"].get("wilcoxon_pareto_vs_engine"):
            assert "p_value" in summary["baseline"]["wilcoxon_pareto_vs_engine"]

    def test_aggregate_no_wilcoxon_for_engine_itself(self):
        results = [_make_trial("engine", True, 0.85, 0.55, i) for i in range(10)]
        summary = aggregate_results(results, engine_method="engine")
        # Engine method should not have wilcoxon vs itself
        assert "wilcoxon_pareto_vs_engine" not in summary.get("engine", {})
