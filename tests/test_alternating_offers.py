"""
Tests for the alternating-offers protocol.
All tests use mocked LLM calls so no API key is required.
"""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock

from src.models.utility import StakeholderProfile, UtilityFunction, Issue
from src.models.negotiation import NegotiationStatus, CounterOffer, AgentAction
from src.protocol.alternating_offers import run_alternating_offers, clamp


ISSUES = [
    {"name": "price", "range": [10.0, 100.0]},
    {"name": "volume", "range": [100.0, 5000.0]},
]

MIDPOINT = {"price": 55.0, "volume": 2550.0}


def _make_profile(name: str, reservation: float = 0.3, honesty: float = 1.0) -> StakeholderProfile:
    issues = [Issue(name=i["name"], min_value=i["range"][0], max_value=i["range"][1]) for i in ISSUES]
    return StakeholderProfile(
        name=name,
        role="Party",
        persona="",
        utility_function=UtilityFunction(
            weights={"price": 0.5, "volume": 0.5},
            ideal_values={"price": 50.0, "volume": 2500.0},
            issues=issues,
        ),
        reservation_value=reservation,
        honesty_level=honesty,
    )


class TestAlternatingOffersTermination:
    @patch("src.protocol.alternating_offers.structured_completion")
    def test_agreement_when_all_accept(self, mock_sc):
        """All agents accepting → outcome.agreement_reached = True."""
        profiles = [_make_profile("A", reservation=0.0), _make_profile("B", reservation=0.0)]

        # Both agents accept on round 1
        mock_sc.return_value = CounterOffer(
            agent_name="A",
            action=AgentAction.ACCEPT,
            proposal=MIDPOINT,
            reasoning="Fine with me",
        )

        outcome = run_alternating_offers(profiles, ISSUES, max_rounds=5)
        assert outcome.agreement_reached is True
        assert outcome.status == NegotiationStatus.AGREED
        assert outcome.protocol_used == "alternating_offers"

    @patch("src.protocol.alternating_offers.structured_completion")
    def test_walk_away_terminates_as_impasse(self, mock_sc):
        """Any agent walking away → outcome.status = IMPASSE, agreement_reached = False."""
        profiles = [_make_profile("A"), _make_profile("B")]

        mock_sc.return_value = CounterOffer(
            agent_name="B",
            action=AgentAction.WALK_AWAY,
            proposal={},
            reasoning="Not going to work",
        )

        outcome = run_alternating_offers(profiles, ISSUES, max_rounds=5)
        assert outcome.agreement_reached is False
        assert outcome.status == NegotiationStatus.IMPASSE

    @patch("src.protocol.alternating_offers.structured_completion")
    def test_max_rounds_impasse(self, mock_sc):
        """Persistent counteroffers with no acceptance → MAX_ROUNDS impasse."""
        profiles = [_make_profile("A", reservation=0.99), _make_profile("B", reservation=0.99)]

        # Always counteroffer, never accept (reservation too high to ever satisfy)
        mock_sc.return_value = CounterOffer(
            agent_name="responder",
            action=AgentAction.COUNTEROFFER,
            proposal={"price": 80.0, "volume": 1000.0},
            reasoning="Not satisfied",
        )

        outcome = run_alternating_offers(profiles, ISSUES, max_rounds=3)
        assert outcome.agreement_reached is False
        assert outcome.status in (NegotiationStatus.MAX_ROUNDS, NegotiationStatus.IMPASSE)
        assert outcome.rounds_taken <= 3

    def test_protocol_used_field_set(self):
        """protocol_used should always be 'alternating_offers'."""
        with patch("src.protocol.alternating_offers.structured_completion") as mock_sc:
            mock_sc.return_value = CounterOffer(
                agent_name="B",
                action=AgentAction.WALK_AWAY,
                proposal={},
                reasoning="No deal",
            )
            profiles = [_make_profile("A"), _make_profile("B")]
            outcome = run_alternating_offers(profiles, ISSUES, max_rounds=2)
            assert outcome.protocol_used == "alternating_offers"

    def test_per_agent_utilities_present(self):
        """Outcome should always have per_agent_utilities populated."""
        with patch("src.protocol.alternating_offers.structured_completion") as mock_sc:
            mock_sc.return_value = CounterOffer(
                agent_name="B",
                action=AgentAction.WALK_AWAY,
                proposal={},
                reasoning="No deal",
            )
            profiles = [_make_profile("A"), _make_profile("B")]
            outcome = run_alternating_offers(profiles, ISSUES, max_rounds=2)
            assert "A" in outcome.per_agent_utilities
            assert "B" in outcome.per_agent_utilities


class TestClampHelper:
    def test_clamp_within_range(self):
        """Values within range should not be changed."""
        result = clamp({"price": 50.0, "volume": 1000.0}, ISSUES)
        assert result["price"] == 50.0
        assert result["volume"] == 1000.0

    def test_clamp_above_max(self):
        result = clamp({"price": 200.0, "volume": 9999.0}, ISSUES)
        assert result["price"] == 100.0
        assert result["volume"] == 5000.0

    def test_clamp_below_min(self):
        result = clamp({"price": -5.0, "volume": 0.0}, ISSUES)
        assert result["price"] == 10.0
        assert result["volume"] == 100.0


class TestRunNegotiationProtocolParam:
    @patch("src.protocol.alternating_offers.structured_completion")
    def test_run_negotiation_routes_to_ao(self, mock_sc):
        """run_negotiation with protocol='alternating_offers' should delegate to AO."""
        from src.protocol.graph import run_negotiation
        mock_sc.return_value = CounterOffer(
            agent_name="B",
            action=AgentAction.WALK_AWAY,
            proposal={},
            reasoning="No",
        )
        profiles = [_make_profile("A"), _make_profile("B")]
        outcome = run_negotiation(
            profiles=profiles,
            issues=ISSUES,
            max_rounds=2,
            protocol="alternating_offers",
        )
        assert outcome.protocol_used == "alternating_offers"
