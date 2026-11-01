"""
Tests for Individual Rationality guarantees across both protocols.
Mocks the LLM to verify that agents and protocols cannot accept proposals below reservation value.
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
)
from src.agents.stakeholder import StakeholderAgent
from src.agents.strategic_stakeholder import StrategicStakeholderAgent
from src.protocol.alternating_offers import run_alternating_offers
from src.protocol.graph import run_negotiation


ISSUES = [
    {"name": "price", "range": [0.0, 100.0]},
]


def _make_profile(name: str, ideal: float = 100.0, reservation: float = 0.9, honesty: float = 1.0) -> StakeholderProfile:
    issues = [Issue(name="price", min_value=0.0, max_value=100.0)]
    return StakeholderProfile(
        name=name,
        role="Party",
        persona="",
        utility_function=UtilityFunction(
            weights={"price": 1.0},
            ideal_values={"price": ideal},
            issues=issues,
        ),
        reservation_value=reservation,
        honesty_level=honesty,
    )


class TestIndividualRationalityAgents:
    @patch("src.agents.stakeholder.structured_completion")
    def test_stakeholder_rejects_sub_reservation_even_if_llm_accepts(self, mock_sc):
        """Mocked LLM claims acceptable=True at utility ~0.1 with reservation 0.9 -> critique.acceptable must be False."""
        profile = _make_profile("HonestAgent", ideal=100.0, reservation=0.9)
        agent = StakeholderAgent(profile)
        # Proposal at 10.0 gives utility = 1 - |10 - 100|/100 = 0.1
        proposal = {"price": 10.0}
        assert agent.evaluate_proposal(proposal) == pytest.approx(0.1, abs=0.01)

        mock_sc.return_value = Critique(
            agent_name="HonestAgent",
            satisfaction_score=1.0,
            acceptable=True,  # LLM erroneously claims acceptable
            issues_to_improve=[],
            desired_directions={},
            concession_willingness=0.5,
            reasoning="I accept anyway",
        )

        critique = agent.generate_critique(proposal, round_number=1, max_rounds=5)
        assert critique.acceptable is False

    @patch("src.agents.strategic_stakeholder.structured_completion")
    def test_strategic_stakeholder_rejects_sub_reservation_even_if_llm_accepts(self, mock_sc):
        """Strategic stakeholder: LLM claims acceptable=True at utility 0.1 with reservation 0.9 -> critique.acceptable must be False."""
        profile = _make_profile("StrategicAgent", ideal=100.0, reservation=0.9, honesty=0.5)
        agent = StrategicStakeholderAgent(profile)
        proposal = {"price": 10.0}
        assert agent.evaluate_proposal(proposal) == pytest.approx(0.1, abs=0.01)

        mock_sc.return_value = Critique(
            agent_name="StrategicAgent",
            satisfaction_score=1.0,
            acceptable=True,
            issues_to_improve=[],
            desired_directions={},
            concession_willingness=0.5,
            reasoning="Strategic LLM hallucinated acceptance",
        )

        critique = agent.generate_critique(proposal, round_number=1, max_rounds=5)
        assert critique.acceptable is False


class TestIndividualRationalityProtocols:
    @patch("src.protocol.alternating_offers.structured_completion")
    def test_alternating_offers_rejects_sub_reservation_accept(self, mock_sc):
        """In alternating offers, if LLM returns ACCEPT below reservation, treat it as counteroffer."""
        # Initial offer is midpoint price=50.
        # Party A ideal=100, reservation=0.9 -> utility at 50 is 0.5 < 0.9.
        # Party B ideal=0, reservation=0.1.
        profiles = [
            _make_profile("AgentA", ideal=100.0, reservation=0.9),
            _make_profile("AgentB", ideal=0.0, reservation=0.9),
        ]

        # LLM returns ACCEPT despite utility 0.5 < 0.9
        mock_sc.return_value = CounterOffer(
            agent_name="AgentA",
            action=AgentAction.ACCEPT,
            proposal={"price": 50.0},
            reasoning="I accept below reservation",
        )

        outcome = run_alternating_offers(profiles, ISSUES, max_rounds=2)
        # Should NOT agree
        assert outcome.agreement_reached is False
        assert outcome.status != NegotiationStatus.AGREED

    @patch("src.protocol.alternating_offers.structured_completion")
    def test_alternating_offers_allows_counteroffer_above_reservation(self, mock_sc):
        """Agents must be able to counteroffer for a better deal even when would_accept is True."""
        profiles = [
            _make_profile("AgentA", ideal=100.0, reservation=0.2),
            _make_profile("AgentB", ideal=0.0, reservation=0.2),
        ]

        # At midpoint 50, utility is 0.5 >= 0.2. But agent wants to COUNTEROFFER.
        mock_sc.return_value = CounterOffer(
            agent_name="AgentA",
            action=AgentAction.COUNTEROFFER,
            proposal={"price": 80.0},
            reasoning="I want more",
        )

        outcome = run_alternating_offers(profiles, ISSUES, max_rounds=1)
        assert outcome.agreement_reached is False

    @patch("src.agents.mediator.structured_completion")
    @patch("src.agents.stakeholder.structured_completion")
    def test_single_text_protocol_no_agreement_if_below_reservation(self, mock_stakeholder_sc, mock_mediator_sc):
        """In single-text protocol, mediator declaring agreement when agent is below reservation must terminate as IMPASSE."""
        profiles = [
            _make_profile("AgentA", ideal=100.0, reservation=0.9),
            _make_profile("AgentB", ideal=0.0, reservation=0.1),
        ]

        # Proposal at 50 gives AgentA utility 0.5 < 0.9
        mock_mediator_sc.side_effect = [
            # Initial propose
            MediatorResponse(
                action=MediatorAction.PROPOSE,
                revised_proposal={"price": 50.0},
                reasoning="Initial",
            ),
            # Mediator declares agreement prematurely
            MediatorResponse(
                action=MediatorAction.DECLARE_AGREEMENT,
                revised_proposal={"price": 50.0},
                reasoning="I declare agreement",
            ),
        ]

        mock_stakeholder_sc.return_value = Critique(
            agent_name="AgentA",
            satisfaction_score=5.0,
            acceptable=True,
            reasoning="ok",
        )

        outcome = run_negotiation(
            profiles=profiles,
            issues=ISSUES,
            max_rounds=2,
            protocol="single_text",
        )

        assert outcome.agreement_reached is False
        assert outcome.status == NegotiationStatus.IMPASSE
