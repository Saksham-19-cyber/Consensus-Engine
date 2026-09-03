"""
Tests for strategic misrepresentation agent behaviour.
All tests use mocked LLM calls so no API key is required.
"""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock

from src.models.utility import StakeholderProfile, UtilityFunction, Issue
from src.models.negotiation import Critique
from src.agents.stakeholder import StakeholderAgent
from src.agents.strategic_stakeholder import StrategicStakeholderAgent
from src.llm.prompts import build_strategic_aggressiveness_instruction


ISSUES = [
    Issue(name="price", min_value=0.0, max_value=100.0),
    Issue(name="volume", min_value=0.0, max_value=1000.0),
]

PROPOSAL = {"price": 50.0, "volume": 500.0}


def _make_profile(honesty_level: float = 1.0) -> StakeholderProfile:
    return StakeholderProfile(
        name="TestAgent",
        role="Tester",
        persona="Test persona",
        utility_function=UtilityFunction(
            weights={"price": 0.6, "volume": 0.4},
            ideal_values={"price": 20.0, "volume": 800.0},
            issues=ISSUES,
        ),
        reservation_value=0.3,
        honesty_level=honesty_level,
        strategic_bias={"price": 20.0, "volume": -100.0} if honesty_level < 1.0 else {},
    )


def _mock_critique(concession_willingness: float = 0.5, satisfaction: float = 5.0) -> Critique:
    return Critique(
        agent_name="TestAgent",
        satisfaction_score=satisfaction,
        acceptable=False,
        issues_to_improve=["price"],
        desired_directions={"price": "lower"},
        concession_willingness=concession_willingness,
        reasoning="Test reasoning",
        round_number=1,
    )


class TestAggressivenessInstruction:
    def test_honest_agent_returns_empty(self):
        result = build_strategic_aggressiveness_instruction(1.0, {})
        assert result == ""

    def test_strategic_agent_returns_instruction(self):
        result = build_strategic_aggressiveness_instruction(0.5, {"price": 10.0})
        assert "STRATEGIC NEGOTIATION MODE" in result
        assert "anchor" in result.lower() or "Anchor" in result

    def test_hard_negotiator_threshold(self):
        result = build_strategic_aggressiveness_instruction(0.3, {})
        assert "HARD negotiator" in result

    def test_bias_description_included(self):
        result = build_strategic_aggressiveness_instruction(0.5, {"price": 15.0, "volume": -50.0})
        assert "price" in result
        assert "volume" in result


class TestStrategicAgentInstantiation:
    def test_honest_profile_creates_regular_agent(self):
        profile = _make_profile(honesty_level=1.0)
        assert not profile.is_strategic
        agent = StakeholderAgent(profile)
        assert isinstance(agent, StakeholderAgent)

    def test_strategic_profile_is_flagged(self):
        profile = _make_profile(honesty_level=0.5)
        assert profile.is_strategic

    def test_strategic_agent_created(self):
        profile = _make_profile(honesty_level=0.5)
        agent = StrategicStakeholderAgent(profile)
        assert isinstance(agent, StrategicStakeholderAgent)


class TestStrategicAgentBehavior:
    @patch("src.agents.strategic_stakeholder.structured_completion")
    def test_acceptable_uses_true_utility(self, mock_completion):
        """The acceptable flag must reflect TRUE utility, not stated utility."""
        profile = _make_profile(honesty_level=0.3)
        agent = StrategicStakeholderAgent(profile)
        true_utility = agent.evaluate_proposal(PROPOSAL)

        # LLM says not acceptable, but true utility >= reservation_value → should be acceptable
        mock_response = _mock_critique(concession_willingness=0.1, satisfaction=2.0)
        mock_response.acceptable = False
        mock_completion.return_value = mock_response

        critique = agent.generate_critique(PROPOSAL, round_number=1, max_rounds=10)

        if true_utility >= profile.reservation_value:
            assert critique.acceptable is True, (
                f"Expected acceptable=True (true_utility={true_utility} >= {profile.reservation_value}) "
                f"but got {critique.acceptable}"
            )

    @patch("src.agents.strategic_stakeholder.structured_completion")
    def test_strategic_agent_suppresses_concession_early(self, mock_completion):
        """In early rounds, strategic agent should cap concession willingness below raw LLM value."""
        profile = _make_profile(honesty_level=0.3)
        agent = StrategicStakeholderAgent(profile)

        # LLM returns high concession willingness
        mock_response = _mock_critique(concession_willingness=0.9, satisfaction=3.0)
        mock_response.acceptable = False
        mock_completion.return_value = mock_response

        critique = agent.generate_critique(PROPOSAL, round_number=1, max_rounds=10)

        # Strategic agent with honesty_level=0.3 in round 1/10 should cap concession significantly
        assert critique.concession_willingness < 0.9, (
            f"Expected suppressed concession in early round, got {critique.concession_willingness}"
        )

    @patch("src.agents.stakeholder.structured_completion")
    def test_honest_agent_preserves_concession(self, mock_completion):
        """Honest agent (honesty_level=1.0) should not suppress concession willingness."""
        profile = _make_profile(honesty_level=1.0)
        agent = StakeholderAgent(profile)

        mock_response = _mock_critique(concession_willingness=0.8, satisfaction=6.0)
        mock_response.acceptable = False
        mock_completion.return_value = mock_response

        critique = agent.generate_critique(PROPOSAL, round_number=1, max_rounds=10)
        assert critique.concession_willingness == 0.8

    def test_stated_utility_understated_for_strategic(self):
        """_compute_stated_utility should return < true_utility for strategic agents in early rounds."""
        profile = _make_profile(honesty_level=0.3)
        agent = StrategicStakeholderAgent(profile)
        true_utility = agent.evaluate_proposal(PROPOSAL)
        stated = agent._compute_stated_utility(true_utility, round_number=1, max_rounds=10)
        assert stated <= true_utility

    def test_stated_utility_unchanged_for_honest(self):
        """_compute_stated_utility should return true_utility for honesty_level=1.0."""
        profile = _make_profile(honesty_level=1.0)
        agent = StrategicStakeholderAgent(profile)
        true_utility = agent.evaluate_proposal(PROPOSAL)
        stated = agent._compute_stated_utility(true_utility, round_number=1, max_rounds=10)
        assert stated == true_utility


class TestMediatorBluffDetection:
    def test_detect_bluffing_flags_low_sat_low_concession(self):
        from src.agents.mediator import MediatorAgent
        mediator = MediatorAgent(["A", "B"], [])
        # Inject 3 rounds of low satisfaction + low concession for agent A
        for _ in range(3):
            mediator._critique_history["A"].append((2.0, 0.05))  # low sat, low concession
            mediator._critique_history["B"].append((7.0, 0.60))  # normal

        suspects = mediator.detect_bluffing()
        assert "A" in suspects
        assert "B" not in suspects

    def test_detect_bluffing_empty_with_too_few_rounds(self):
        from src.agents.mediator import MediatorAgent
        mediator = MediatorAgent(["A"], [])
        mediator._critique_history["A"].append((1.0, 0.01))  # Only 1 round — not enough

        suspects = mediator.detect_bluffing()
        assert suspects == []


class TestStrategicScenario:
    def test_scenario_generates_with_honesty_levels(self):
        from scenarios.strategic_negotiation import StrategicNegotiationScenario
        scenario = StrategicNegotiationScenario()
        profiles, issues = scenario.generate(seed=42)

        assert len(profiles) == 3
        assert len(issues) == 5
        for p in profiles:
            assert 0.3 <= p.honesty_level <= 1.0, f"honesty_level out of range: {p.honesty_level}"

    def test_generate_honest_sets_all_to_one(self):
        from scenarios.strategic_negotiation import StrategicNegotiationScenario
        scenario = StrategicNegotiationScenario()
        profiles, _ = scenario.generate_honest(seed=42)
        for p in profiles:
            assert p.honesty_level == 1.0

    def test_generate_all_strategic_sets_fixed_honesty(self):
        from scenarios.strategic_negotiation import StrategicNegotiationScenario
        scenario = StrategicNegotiationScenario()
        profiles, _ = scenario.generate_all_strategic(seed=42, honesty=0.3)
        for p in profiles:
            assert p.honesty_level == 0.3
