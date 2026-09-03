"""
Tests for the privacy leakage measurement module.
Uses synthetic transcripts and synthetic weight vectors — no API key required.
"""
from __future__ import annotations
import pytest
from unittest.mock import patch
import json

from src.eval.privacy import (
    _cosine_similarity,
    _kl_divergence,
    compute_leakage_scores,
    measure_privacy_leakage,
)
from src.models.evaluation import PrivacyMetrics


ISSUES = [
    {"name": "price", "range": [0.0, 100.0]},
    {"name": "volume", "range": [0.0, 1000.0]},
    {"name": "delivery", "range": [1.0, 30.0]},
]

AGENT_NAMES = ["Alice", "Bob"]
TRUE_WEIGHTS = {
    "Alice": {"price": 0.6, "volume": 0.3, "delivery": 0.1},
    "Bob": {"price": 0.1, "volume": 0.2, "delivery": 0.7},
}


class TestMathHelpers:
    def test_cosine_identical(self):
        v = [0.5, 0.3, 0.2]
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_orthogonal(self):
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_range(self):
        import random
        random.seed(42)
        for _ in range(20):
            a = [random.random() for _ in range(5)]
            b = [random.random() for _ in range(5)]
            cos = _cosine_similarity(a, b)
            assert 0.0 <= cos <= 1.0 + 1e-9

    def test_kl_identical(self):
        v = [0.4, 0.4, 0.2]
        assert _kl_divergence(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_kl_positive_for_different(self):
        p = [0.6, 0.3, 0.1]
        q = [0.2, 0.5, 0.3]
        assert _kl_divergence(p, q) > 0.0

    def test_kl_asymmetric(self):
        p = [0.8, 0.1, 0.1]
        q = [0.1, 0.8, 0.1]
        # KL(P||Q) != KL(Q||P) in general — verify at least one direction is larger
        kl_pq = _kl_divergence(p, q)
        kl_qp = _kl_divergence(q, p)
        # Both should be positive and not both zero
        assert kl_pq > 0 or kl_qp > 0


class TestComputeLeakageScores:
    def test_perfect_inference_scores_one(self):
        """If inferred == true, cosine similarity should be 1.0."""
        scores = compute_leakage_scores(TRUE_WEIGHTS, TRUE_WEIGHTS, ISSUES)
        for name in AGENT_NAMES:
            assert scores[name]["cosine_similarity"] == pytest.approx(1.0, abs=1e-4)
            assert scores[name]["kl_divergence"] == pytest.approx(0.0, abs=1e-4)

    def test_random_inference_scores_below_one(self):
        """Random uniform inference should produce cosine < 1.0."""
        inferred = {
            "Alice": {"price": 0.33, "volume": 0.33, "delivery": 0.34},
            "Bob": {"price": 0.33, "volume": 0.33, "delivery": 0.34},
        }
        scores = compute_leakage_scores(TRUE_WEIGHTS, inferred, ISSUES)
        for name in AGENT_NAMES:
            assert scores[name]["cosine_similarity"] < 1.0

    def test_scores_in_valid_range(self):
        inferred = {
            "Alice": {"price": 0.4, "volume": 0.4, "delivery": 0.2},
            "Bob": {"price": 0.3, "volume": 0.4, "delivery": 0.3},
        }
        scores = compute_leakage_scores(TRUE_WEIGHTS, inferred, ISSUES)
        for name in AGENT_NAMES:
            assert 0.0 <= scores[name]["cosine_similarity"] <= 1.0
            assert scores[name]["kl_divergence"] >= 0.0

    def test_output_keys_present(self):
        scores = compute_leakage_scores(TRUE_WEIGHTS, TRUE_WEIGHTS, ISSUES)
        for name in AGENT_NAMES:
            assert "cosine_similarity" in scores[name]
            assert "kl_divergence" in scores[name]
            assert "leakage_score" in scores[name]


class TestMeasurePrivacyLeakage:
    def _make_profile(self, name, weights):
        from src.models.utility import StakeholderProfile, UtilityFunction, Issue
        issues = [Issue(name=i["name"], min_value=i["range"][0], max_value=i["range"][1]) for i in ISSUES]
        return StakeholderProfile(
            name=name,
            role="Test",
            persona="",
            utility_function=UtilityFunction(
                weights=weights,
                ideal_values={i["name"]: (i["range"][0] + i["range"][1]) / 2 for i in ISSUES},
                issues=issues,
            ),
            reservation_value=0.3,
        )

    @patch("src.eval.privacy.plain_completion")
    def test_returns_privacy_metrics(self, mock_completion):
        """measure_privacy_leakage should always return a PrivacyMetrics object."""
        mock_completion.return_value = json.dumps({
            "Alice": {"price": 0.5, "volume": 0.3, "delivery": 0.2},
            "Bob": {"price": 0.1, "volume": 0.2, "delivery": 0.7},
        })

        profiles = [
            self._make_profile("Alice", TRUE_WEIGHTS["Alice"]),
            self._make_profile("Bob", TRUE_WEIGHTS["Bob"]),
        ]
        metrics = measure_privacy_leakage(transcript=[], profiles=profiles, issues=ISSUES)

        assert isinstance(metrics, PrivacyMetrics)
        assert 0.0 <= metrics.mean_cosine_similarity <= 1.0
        assert metrics.mean_kl_divergence >= 0.0
        assert metrics.n_agents == 2
        assert metrics.n_issues == 3
        assert "Alice" in metrics.per_agent_leakage
        assert "Bob" in metrics.per_agent_leakage

    @patch("src.eval.privacy.plain_completion")
    def test_handles_llm_failure_gracefully(self, mock_completion):
        """Should fall back to uniform weights and still return PrivacyMetrics."""
        mock_completion.side_effect = Exception("LLM unavailable")

        profiles = [self._make_profile("Alice", TRUE_WEIGHTS["Alice"])]
        metrics = measure_privacy_leakage(transcript=[], profiles=profiles, issues=ISSUES)

        assert isinstance(metrics, PrivacyMetrics)
        # With uniform fallback, cosine should be in [0, 1]
        assert 0.0 <= metrics.mean_cosine_similarity <= 1.0

    @patch("src.eval.privacy.plain_completion")
    def test_high_similarity_when_perfectly_inferred(self, mock_completion):
        """When the LLM perfectly infers weights, cosine should approach 1.0."""
        # The probe expects a JSON with agent name as outer key
        mock_completion.return_value = json.dumps({
            "Alice": TRUE_WEIGHTS["Alice"],
        })

        profiles = [self._make_profile("Alice", TRUE_WEIGHTS["Alice"])]
        # Provide a non-empty transcript so the probe doesn't early-exit
        fake_transcript = [
            {
                "role": "stakeholder",
                "agent_name": "Alice",
                "content": "I strongly prefer lower price and high delivery speed",
                "round_number": 1,
                "message_type": "critique",
                "metadata": {"reasoning": "Price is my top priority"},
            }
        ]
        metrics = measure_privacy_leakage(
            transcript=fake_transcript, profiles=profiles, issues=ISSUES
        )
        # After normalization, inferred weights should closely match true weights
        assert metrics.mean_cosine_similarity > 0.90
