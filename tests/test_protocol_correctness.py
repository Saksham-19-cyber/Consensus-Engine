"""
Tests for Phase 2: Protocol Correctness
- alternating_offers proposer acceptance requirement and round-robin proposer rotation
- graph.py mediator agreement/impasse direct termination, rounds_taken fix, and max_rounds pass-through
- strategic_negotiation buyer quality_tier inflating HIGHER
"""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock

from src.models.utility import StakeholderProfile, UtilityFunction, Issue
from src.models.negotiation import (
    Critique,
    CounterOffer,
    AgentAction,
    MediatorResponse,
    MediatorAction,
    NegotiationStatus,
    NegotiationOutcome,
)
from src.protocol.alternating_offers import run_alternating_offers
from src.protocol.graph import run_negotiation, build_negotiation_graph
from scenarios.strategic_negotiation import StrategicNegotiationScenario


ISSUES = [
    {"name": "price", "range": [0.0, 100.0]},
    {"name": "volume", "range": [10.0, 1000.0]},
]


def _make_profile(
    name: str,
    ideal_price: float = 100.0,
    ideal_volume: float = 500.0,
    reservation: float = 0.3,
) -> StakeholderProfile:
    issues = [
        Issue(name="price", min_value=0.0, max_value=100.0),
        Issue(name="volume", min_value=10.0, max_value=1000.0),
    ]
    return StakeholderProfile(
        name=name,
        role="Party",
        persona="",
        utility_function=UtilityFunction(
            weights={"price": 0.5, "volume": 0.5},
            ideal_values={"price": ideal_price, "volume": ideal_volume},
            issues=issues,
        ),
        reservation_value=reservation,
        honesty_level=1.0,
    )


class TestAlternatingOffersCorrectness:
    @patch("src.protocol.alternating_offers.structured_completion")
    def test_proposer_acceptance_required(self, mock_sc):
        """Even if all responders accept, if the proposer does not accept the current proposal, agreement is NOT reached."""
        # Midpoint: price=50, volume=505.
        # Party A: ideal_price=100, ideal_volume=1000, reservation=0.9 (utility at midpoint ~0.5 < 0.9 -> does NOT accept)
        # Party B: reservation=0.0 (utility ~0.5 >= 0.0 -> accepts)
        profiles = [
            _make_profile("PartyA", ideal_price=100.0, ideal_volume=1000.0, reservation=0.9),
            _make_profile("PartyB", ideal_price=50.0, ideal_volume=505.0, reservation=0.0),
        ]

        # Force PartyA to be the first offerer
        with patch("random.choice", return_value="PartyA"):
            # PartyB responds with ACCEPT
            mock_sc.return_value = CounterOffer(
                agent_name="PartyB",
                action=AgentAction.ACCEPT,
                proposal={"price": 50.0, "volume": 505.0},
                reasoning="I accept",
            )

            outcome = run_alternating_offers(profiles, ISSUES, max_rounds=2)
            # PartyA does not accept the current proposal, so agreement must NOT be reached
            assert outcome.agreement_reached is False
            assert outcome.status != NegotiationStatus.AGREED

    @patch("src.protocol.alternating_offers.structured_completion")
    def test_proposers_rotate_round_robin(self, mock_sc):
        """Proposers must rotate round-robin across rounds."""
        profiles = [
            _make_profile("Agent1"),
            _make_profile("Agent2"),
            _make_profile("Agent3"),
        ]

        proposers_recorded = []

        def mock_completion(messages, **kwargs):
            # Record who the current proposer is by inspecting user prompt
            user_msg = messages[1]["content"]
            # "Current proposal (offered by <proposer>):"
            for line in user_msg.splitlines():
                if "Current proposal (offered by" in line:
                    proposer = line.split("offered by ")[1].rstrip("):")
                    proposers_recorded.append(proposer)
                    break
            return CounterOffer(
                agent_name="responder",
                action=AgentAction.COUNTEROFFER,
                proposal={"price": 60.0, "volume": 600.0},
                reasoning="counter",
            )

        mock_sc.side_effect = mock_completion

        with patch("random.choice", return_value="Agent1"):
            outcome = run_alternating_offers(profiles, ISSUES, max_rounds=3)

        # In round 1, Agent1 is proposer -> Agent2 and Agent3 are prompted with proposer Agent1
        # In round 2, proposer should rotate to Agent2 -> responders prompted with proposer Agent2
        # In round 3, proposer should rotate to Agent3 -> responders prompted with proposer Agent3
        unique_proposers_in_order = []
        for p in proposers_recorded:
            if not unique_proposers_in_order or unique_proposers_in_order[-1] != p:
                unique_proposers_in_order.append(p)

        assert unique_proposers_in_order == ["Agent1", "Agent2", "Agent3"]


class TestGraphProtocolCorrectness:
    @patch("src.agents.mediator.structured_completion")
    @patch("src.agents.stakeholder.structured_completion")
    def test_mediator_declaration_terminates_directly_and_rounds_taken(
        self, mock_stakeholder_sc, mock_mediator_sc
    ):
        """When mediator declares agreement on round 1 critiques, no round 2 critiques run, and rounds_taken == 1."""
        profiles = [
            _make_profile("A", ideal_price=50.0, ideal_volume=505.0, reservation=0.1),
            _make_profile("B", ideal_price=50.0, ideal_volume=505.0, reservation=0.1),
        ]

        mock_mediator_sc.side_effect = [
            # Initial proposal
            MediatorResponse(
                action=MediatorAction.PROPOSE,
                revised_proposal={"price": 50.0, "volume": 505.0},
                reasoning="Initial",
            ),
            # After round 1 critiques, mediator declares agreement
            MediatorResponse(
                action=MediatorAction.DECLARE_AGREEMENT,
                revised_proposal={"price": 50.0, "volume": 505.0},
                reasoning="Agreed",
            ),
        ]

        stakeholder_call_count = 0

        def mock_stakeholder(*args, **kwargs):
            nonlocal stakeholder_call_count
            stakeholder_call_count += 1
            return Critique(
                agent_name="A",
                satisfaction_score=9.0,
                acceptable=True,
                reasoning="Great deal",
            )

        mock_stakeholder_sc.side_effect = mock_stakeholder

        outcome = run_negotiation(
            profiles=profiles,
            issues=ISSUES,
            max_rounds=5,
            protocol="single_text",
        )

        assert outcome.agreement_reached is True
        assert outcome.status == NegotiationStatus.AGREED
        assert outcome.rounds_taken == 1
        # Exactly 2 stakeholder calls (one for A, one for B in round 1 only)
        assert stakeholder_call_count == 2

    @patch("src.agents.mediator.structured_completion")
    @patch("src.agents.stakeholder.structured_completion")
    def test_mediator_uses_run_max_rounds(
        self, mock_stakeholder_sc, mock_mediator_sc
    ):
        """Mediator prompts should use the run's max_rounds, not settings.max_rounds."""
        profiles = [
            _make_profile("A", reservation=0.1),
            _make_profile("B", reservation=0.1),
        ]

        captured_system_prompts = []

        def mock_mediator(messages, **kwargs):
            captured_system_prompts.append(messages[0]["content"])
            return MediatorResponse(
                action=MediatorAction.DECLARE_AGREEMENT,
                revised_proposal={"price": 50.0, "volume": 505.0},
                reasoning="Agreed",
            )

        mock_mediator_sc.side_effect = mock_mediator
        mock_stakeholder_sc.return_value = Critique(
            agent_name="A",
            satisfaction_score=9.0,
            acceptable=True,
            reasoning="Ok",
        )

        custom_max_rounds = 4
        outcome = run_negotiation(
            profiles=profiles,
            issues=ISSUES,
            max_rounds=custom_max_rounds,
            protocol="single_text",
        )

        # Confirm that custom_max_rounds (4) is present in the mediator system prompt
        assert len(captured_system_prompts) > 0
        assert f"/{custom_max_rounds}" in captured_system_prompts[0]


class TestBuyerStrategicBias:
    def test_buyer_quality_tier_inflates_higher(self):
        """Buyer's quality_tier bias direction must inflate HIGHER (positive offset), while delivery_days is lower."""
        scenario = StrategicNegotiationScenario()
        for seed in range(10):
            profiles, _ = scenario.generate(seed=seed)
            buyer = next(p for p in profiles if p.role == "Buyer")
            if buyer.honesty_level < 1.0:
                assert "quality_tier" in buyer.strategic_bias
                assert buyer.strategic_bias["quality_tier"] > 0, (
                    f"seed={seed}: buyer quality_tier bias must be > 0 (higher), got {buyer.strategic_bias['quality_tier']}"
                )
                assert buyer.strategic_bias["delivery_days"] < 0, (
                    f"seed={seed}: buyer delivery_days bias must be < 0 (lower), got {buyer.strategic_bias['delivery_days']}"
                )
