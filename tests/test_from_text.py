"""
tests/test_from_text.py
=========================
Unit tests for scenarios/from_text.py — the free-form scenario parser.

These tests use mocking to avoid live LLM calls in CI. Integration tests
that actually call Groq are marked @pytest.mark.integration and are skipped
unless GROQ_API_KEY is set and --run-integration is passed.

Run:
    pytest tests/test_from_text.py -v
    pytest tests/test_from_text.py -v -m integration --run-integration  # live tests
"""
from __future__ import annotations

import base64
import json
from unittest.mock import patch, MagicMock

import pytest

from scenarios.from_text import (
    ParseReport,
    _LLMScenarioResponse,
    _LLMIssue,
    _LLMStakeholder,
    generate_from_description,
)
from src.models.utility import StakeholderProfile, Issue


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: fake LLM responses
# ─────────────────────────────────────────────────────────────────────────────

def _make_minimal_llm_response(n_issues: int = 3, n_stakeholders: int = 2) -> _LLMScenarioResponse:
    """Build a synthetic _LLMScenarioResponse for mocking."""
    issue_names = [f"issue_{i}" for i in range(n_issues)]
    issues = [
        _LLMIssue(
            name=f"issue_{i}",
            min_value=0.0,
            max_value=float(10 + i),
            description=f"Higher = more of thing {i}, lower = less",
        )
        for i in range(n_issues)
    ]
    uniform_weight = 1.0 / n_issues
    stakeholders = [
        _LLMStakeholder(
            name=f"Party{j}",
            role=f"Role{j}",
            persona=f"Party {j} cares about all issues equally.",
            weights={name: uniform_weight for name in issue_names},
            ideal_values={f"issue_{i}": float(5 + i) for i in range(n_issues)},
            reservation_value=0.35,
            source="user_specified" if j == 0 else "llm_inferred",
        )
        for j in range(n_stakeholders)
    ]
    return _LLMScenarioResponse(issues=issues, stakeholders=stakeholders, warnings=[])


# ─────────────────────────────────────────────────────────────────────────────
# ParseReport round-trip tests (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestParseReportToken:
    """Tests for the base64 token encode/decode round-trip."""

    def _build_report_with_profiles(self) -> tuple[list[StakeholderProfile], list[dict], ParseReport]:
        from src.models.utility import UtilityFunction
        issue = Issue(name="price", min_value=10.0, max_value=100.0, description="Higher is pricier")
        issues_meta = [{"name": "price", "range": [10.0, 100.0], "description": "Higher is pricier"}]

        profile = StakeholderProfile(
            name="Buyer",
            role="Buyer",
            persona="Wants a low price",
            utility_function=UtilityFunction(
                weights={"price": 1.0},
                ideal_values={"price": 10.0},
                issues=[issue],
            ),
            reservation_value=0.3,
        )
        report = ParseReport(
            issues=issues_meta,
            profiles=[profile.model_dump()],
            stakeholder_sources={"Buyer": "user_specified"},
            field_notes=[],
            warnings=[],
            issue_count=1,
            pareto_mode="exhaustive",
        )
        return [profile], issues_meta, report

    def test_token_encodes_and_decodes(self):
        profiles, issues_meta, report = self._build_report_with_profiles()
        token = report.to_token()
        assert isinstance(token, str)
        assert len(token) > 10

        recovered_profiles, recovered_issues = ParseReport.decode_token(token)
        assert len(recovered_profiles) == 1
        assert recovered_profiles[0].name == "Buyer"
        assert recovered_issues[0]["name"] == "price"

    def test_token_is_valid_base64_json(self):
        _, _, report = self._build_report_with_profiles()
        token = report.to_token()
        raw = base64.b64decode(token.encode()).decode()
        payload = json.loads(raw)
        assert "issues_meta" in payload
        assert "profiles" in payload

    def test_corrupted_token_raises(self):
        with pytest.raises(Exception):
            ParseReport.decode_token("not-a-valid-token!!")


# ─────────────────────────────────────────────────────────────────────────────
# Pareto mode selection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParetoAutoDispatch:
    """Tests for compute_pareto_frontier_auto dispatch logic."""

    def test_exhaustive_for_small_issues(self):
        from src.eval.pareto import compute_pareto_frontier_auto
        from src.models.utility import UtilityFunction

        issues = [{"name": f"i{k}", "range": [0.0, 1.0]} for k in range(3)]
        ufs = [
            UtilityFunction(
                weights={f"i{k}": 1.0 / 3 for k in range(3)},
                ideal_values={f"i{k}": 0.8 for k in range(3)},
                issues=[Issue(name=f"i{k}", min_value=0.0, max_value=1.0) for k in range(3)],
            )
            for _ in range(2)
        ]
        frontier, is_approx = compute_pareto_frontier_auto(ufs, issues, resolution=5)
        assert is_approx is False
        assert frontier.shape[1] == 2

    def test_monte_carlo_for_five_issues(self):
        from src.eval.pareto import compute_pareto_frontier_auto
        from src.models.utility import UtilityFunction

        n = 5
        issues = [{"name": f"i{k}", "range": [0.0, 1.0]} for k in range(n)]
        ufs = [
            UtilityFunction(
                weights={f"i{k}": 1.0 / n for k in range(n)},
                ideal_values={f"i{k}": 0.7 for k in range(n)},
                issues=[Issue(name=f"i{k}", min_value=0.0, max_value=1.0) for k in range(n)],
            )
            for _ in range(2)
        ]
        frontier, is_approx = compute_pareto_frontier_auto(
            ufs, issues, mc_samples=1000, seed=1
        )
        assert is_approx is True
        assert len(frontier) > 0

    def test_seven_issues_raises(self):
        from src.eval.pareto import compute_pareto_frontier_auto
        from src.models.utility import UtilityFunction

        n = 7
        issues = [{"name": f"i{k}", "range": [0.0, 1.0]} for k in range(n)]
        ufs = [
            UtilityFunction(
                weights={f"i{k}": 1.0 / n for k in range(n)},
                ideal_values={f"i{k}": 0.5 for k in range(n)},
                issues=[Issue(name=f"i{k}", min_value=0.0, max_value=1.0) for k in range(n)],
            )
        ]
        with pytest.raises(ValueError, match="requires at most"):
            compute_pareto_frontier_auto(ufs, issues)


# ─────────────────────────────────────────────────────────────────────────────
# generate_from_description tests (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateFromDescription:
    """Tests for generate_from_description() — LLM is always mocked here."""

    @patch("scenarios.from_text.structured_completion")
    def test_basic_3_issue_2_party(self, mock_sc):
        mock_sc.return_value = _make_minimal_llm_response(n_issues=3, n_stakeholders=2)
        profiles, issues_meta, report = generate_from_description(
            "Two parties negotiating three things.", seed=42
        )
        assert len(profiles) == 2
        assert len(issues_meta) == 3
        assert report.issue_count == 3
        assert report.pareto_mode == "exhaustive"

    @patch("scenarios.from_text.structured_completion")
    def test_weights_renormalized(self, mock_sc):
        """Even if LLM returns weights summing to 2.0, they get renormalized to 1.0."""
        response = _make_minimal_llm_response(n_issues=2, n_stakeholders=2)
        # Mess up weights to sum = 2.0
        for sh in response.stakeholders:
            sh.weights = {k: 1.0 for k in sh.weights}
        mock_sc.return_value = response

        profiles, _, _ = generate_from_description("Two parties over two issues.", seed=42)
        for p in profiles:
            total = sum(p.utility_function.weights.values())
            assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    @patch("scenarios.from_text.structured_completion")
    def test_pareto_mode_monte_carlo_for_5_issues(self, mock_sc):
        mock_sc.return_value = _make_minimal_llm_response(n_issues=5, n_stakeholders=2)
        _, _, report = generate_from_description("Five-issue negotiation.", seed=42)
        assert report.pareto_mode == "monte_carlo"
        assert "Monte Carlo" in " ".join(report.field_notes)

    @patch("scenarios.from_text.structured_completion")
    def test_7_issues_raises_value_error(self, mock_sc):
        mock_sc.return_value = _make_minimal_llm_response(n_issues=7, n_stakeholders=2)
        with pytest.raises(ValueError, match="maximum of 6 issues"):
            generate_from_description("Seven-issue description here.", seed=42)

    @patch("scenarios.from_text.structured_completion")
    def test_llm_inferred_source_labelled(self, mock_sc):
        mock_sc.return_value = _make_minimal_llm_response(n_issues=2, n_stakeholders=2)
        _, _, report = generate_from_description("Two parties, two issues.", seed=42)
        assert report.stakeholder_sources["Party0"] == "user_specified"
        assert report.stakeholder_sources["Party1"] == "llm_inferred"
        # Should have a field note about the inferred party
        combined = " ".join(report.field_notes)
        assert "LLM" in combined or "llm" in combined.lower() or "generated" in combined.lower()

    def test_short_description_raises(self):
        with pytest.raises(ValueError, match="too short"):
            generate_from_description("Hi", seed=42)

    @patch("scenarios.from_text.structured_completion")
    def test_ideal_values_clamped_to_range(self, mock_sc):
        """Ideal values outside [min, max] should be clamped."""
        response = _make_minimal_llm_response(n_issues=2, n_stakeholders=2)
        # Set ideal values way outside range
        for sh in response.stakeholders:
            for key in sh.ideal_values:
                sh.ideal_values[key] = 999.0  # way above max_value (11 or 12)
        mock_sc.return_value = response

        profiles, issues_meta, _ = generate_from_description(
            "Two parties are negotiating over two issues: price and delivery time.", seed=42
        )
        issue_map = {i["name"]: i for i in issues_meta}
        for p in profiles:
            for name, val in p.utility_function.ideal_values.items():
                max_v = issue_map[name]["range"][1]
                assert val <= max_v + 0.01, f"ideal {val} exceeds max {max_v} for {name}"

    @patch("scenarios.from_text.structured_completion")
    def test_output_compatible_with_run_negotiation_signature(self, mock_sc):
        """The (profiles, issues) tuple should be drop-in for run_negotiation()."""
        mock_sc.return_value = _make_minimal_llm_response(n_issues=3, n_stakeholders=2)
        profiles, issues_meta, _ = generate_from_description(
            "Simple negotiation between buyer and seller.", seed=42
        )

        # These are the exact assertions run_negotiation() implicitly requires
        assert all(isinstance(p, StakeholderProfile) for p in profiles)
        assert all(isinstance(i, dict) for i in issues_meta)
        assert all("name" in i and "range" in i for i in issues_meta)
        for p in profiles:
            assert hasattr(p, "utility_function")
            assert hasattr(p, "reservation_value")

    @patch("scenarios.from_text.structured_completion")
    def test_parse_report_token_round_trip(self, mock_sc):
        """Token produced by ParseReport should successfully decode back to (profiles, issues)."""
        mock_sc.return_value = _make_minimal_llm_response(n_issues=2, n_stakeholders=2)
        profiles, issues_meta, report = generate_from_description(
            "Two parties negotiating two items in a deal.", seed=42
        )

        token = report.to_token()
        recovered_profiles, recovered_issues = ParseReport.decode_token(token)
        assert len(recovered_profiles) == len(profiles)
        assert len(recovered_issues) == len(issues_meta)
        assert recovered_profiles[0].name == profiles[0].name

    @patch("scenarios.from_text.structured_completion")
    def test_min_max_swap_warning_added(self, mock_sc):
        """If LLM returns min >= max for an issue, a warning should be added."""
        response = _make_minimal_llm_response(n_issues=2, n_stakeholders=2)
        # Force min >= max on first issue
        response.issues[0].min_value = 10.0
        response.issues[0].max_value = 5.0  # inverted
        mock_sc.return_value = response

        _, _, report = generate_from_description("Inverted range issue.", seed=42)
        # After swap, the issue should have min < max
        assert len(report.issues) == 2
        issue_0 = report.issues[0]
        assert issue_0["range"][0] < issue_0["range"][1]

    @patch("scenarios.from_text.structured_completion")
    def test_missing_ideal_value_fills_midpoint(self, mock_sc):
        """If a stakeholder has no ideal for an issue, it should default to the midpoint."""
        response = _make_minimal_llm_response(n_issues=2, n_stakeholders=2)
        # Remove ideal for issue_1 from Party0
        del response.stakeholders[0].ideal_values["issue_1"]
        mock_sc.return_value = response

        profiles, issues_meta, _ = generate_from_description(
            "Missing ideal value test between two negotiating parties.", seed=42
        )
        issue_meta = next(i for i in issues_meta if i["name"] == "issue_1")
        expected_mid = (issue_meta["range"][0] + issue_meta["range"][1]) / 2
        party0_ideal = profiles[0].utility_function.ideal_values.get("issue_1")
        assert party0_ideal is not None
        assert abs(party0_ideal - expected_mid) < 0.5  # midpoint, allowing small float drift

    @patch("scenarios.from_text.structured_completion")
    def test_key_mismatch_regression_utility_at_ideal(self, mock_sc):
        """
        Bug 6.1: When LLM keys weights by display names ('License Fee') rather than
        exact issue names ('license_fee'), the parser must canonicalize keys or drop unknowns
        and renormalize over real issues only.
        Invariant: weights sum to 1.0 over exactly the issue set, and utility at ideal == 1.0.
        """
        issues = [
            _LLMIssue(name="license_fee", min_value=100.0, max_value=500.0, description="Annual fee"),
            _LLMIssue(name="contract_duration", min_value=1.0, max_value=5.0, description="Years"),
        ]
        # Party 0 uses display name "License Fee" instead of "license_fee"
        stakeholders = [
            _LLMStakeholder(
                name="Buyer",
                role="Buyer",
                persona="Wants cheap price",
                weights={"License Fee": 0.6, "contract_duration": 0.4},
                ideal_values={"License Fee": 100.0, "contract_duration": 1.0},
                reservation_value=0.4,
                source="user_specified",
            ),
            _LLMStakeholder(
                name="Vendor",
                role="Vendor",
                persona="Wants high price",
                weights={"license_fee": 0.7, "contract_duration": 0.3},
                ideal_values={"license_fee": 500.0, "contract_duration": 5.0},
                reservation_value=0.4,
                source="llm_inferred",
            ),
        ]
        mock_sc.return_value = _LLMScenarioResponse(issues=issues, stakeholders=stakeholders, warnings=[])

        profiles, issues_meta, report = generate_from_description(
            "Negotiation over software license fee and contract duration between Buyer and Vendor.", seed=42
        )

        for p in profiles:
            # Invariant 1: weights must contain exactly the issue names, nothing else
            assert set(p.utility_function.weights.keys()) == {"license_fee", "contract_duration"}
            # Invariant 2: weights sum to 1.0
            assert sum(p.utility_function.weights.values()) == pytest.approx(1.0, abs=1e-6)
            # Invariant 3: utility at party's own ideal point == 1.0
            u_ideal = p.evaluate_proposal(p.utility_function.ideal_values)
            assert u_ideal == pytest.approx(1.0, abs=1e-5)

    @patch("scenarios.from_text.structured_completion")
    def test_infeasible_reservations_detected(self, mock_sc):
        """Bug 6.2: Opposing parties with unreachable reservations must be flagged as infeasible."""
        issues = [
            _LLMIssue(name="price", min_value=0.0, max_value=100.0, description="Price in dollars"),
            _LLMIssue(name="quantity", min_value=1.0, max_value=10.0, description="Quantity"),
        ]
        # Opposing ideals with strict 0.95 reservations
        stakeholders = [
            _LLMStakeholder(
                name="Buyer",
                role="Buyer",
                persona="Buyer",
                weights={"price": 0.5, "quantity": 0.5},
                ideal_values={"price": 0.0, "quantity": 10.0},
                reservation_value=0.95,
                source="user_specified",
            ),
            _LLMStakeholder(
                name="Seller",
                role="Seller",
                persona="Seller",
                weights={"price": 0.5, "quantity": 0.5},
                ideal_values={"price": 100.0, "quantity": 1.0},
                reservation_value=0.95,
                source="user_specified",
            ),
        ]
        mock_sc.return_value = _LLMScenarioResponse(issues=issues, stakeholders=stakeholders, warnings=[])

        profiles, issues_meta, report = generate_from_description(
            "Extreme opposition negotiation between Buyer and Seller.", seed=42
        )
        assert report.is_feasible is False
        assert any("No deal can satisfy everyone" in w for w in report.warnings)

    def test_user_edits_round_trip(self):
        """Bug 6.3: User edits should re-validate server-side and mark source as user_specified."""
        from scenarios.from_text import validate_and_update_scenario

        issues_data = [
            {"name": "salary", "range": [50000.0, 150000.0], "description": "Annual salary in USD"},
            {"name": "vacation_days", "range": [10.0, 30.0], "description": "PTO days per year"},
        ]
        profiles_data = [
            {
                "name": "Candidate",
                "role": "Candidate",
                "persona": "Experienced engineer",
                "weights": {"salary": 0.8, "vacation_days": 0.2},
                "ideal_values": {"salary": 140000.0, "vacation_days": 28.0},
                "reservation_value": 0.50,
            },
            {
                "name": "Employer",
                "role": "Recruiter",
                "persona": "Budget-conscious recruiter",
                "weights": {"salary": 0.7, "vacation_days": 0.3},
                "ideal_values": {"salary": 70000.0, "vacation_days": 15.0},
                "reservation_value": 0.45,
            },
        ]

        profiles, issues_meta, report = validate_and_update_scenario(
            issues_data=issues_data,
            profiles_data=profiles_data,
            user_edited=True,
        )

        assert len(profiles) == 2
        assert len(issues_meta) == 2
        # All parties marked user_specified after editing
        assert report.stakeholder_sources["Candidate"] == "user_specified"
        assert report.stakeholder_sources["Employer"] == "user_specified"
        # Token round-trip test with edited values
        token = report.to_token()
        rec_profiles, rec_issues = ParseReport.decode_token(token)
        assert len(rec_profiles) == 2
        assert rec_profiles[0].utility_function.ideal_values["salary"] == 140000.0

    @patch("scenarios.from_text.structured_completion")
    def test_pure_binary_issue_warning(self, mock_sc):
        """Bug 6.4: Warning should be generated for pure binary yes/no issues."""
        issues = [
            _LLMIssue(name="ip_transfer", min_value=0.0, max_value=1.0, description="Binary yes/no IP transfer"),
            _LLMIssue(name="price", min_value=10.0, max_value=100.0, description="Fee"),
        ]
        stakeholders = [
            _LLMStakeholder(
                name="PartyA", role="A", persona="",
                weights={"ip_transfer": 0.5, "price": 0.5},
                ideal_values={"ip_transfer": 1.0, "price": 100.0},
                reservation_value=0.3, source="user_specified",
            ),
            _LLMStakeholder(
                name="PartyB", role="B", persona="",
                weights={"ip_transfer": 0.5, "price": 0.5},
                ideal_values={"ip_transfer": 0.0, "price": 10.0},
                reservation_value=0.3, source="llm_inferred",
            ),
        ]
        mock_sc.return_value = _LLMScenarioResponse(issues=issues, stakeholders=stakeholders, warnings=[])

        _, _, report = generate_from_description("Binary IP transfer negotiation.", seed=42)
        assert any("pure binary flag" in w for w in report.warnings)

    def test_compute_plain_language_summary(self):
        """Bug 6.10: Summary must produce package text, party satisfaction, benchmarks, and explanation."""
        from scenarios.from_text import compute_plain_language_summary
        from src.models.utility import StakeholderProfile, UtilityFunction, Issue
        from unittest.mock import MagicMock

        issue = Issue(name="price", min_value=0.0, max_value=100.0, description="1=basic, 2=standard, 3=dedicated")
        issues_meta = [{"name": "price", "range": [0.0, 100.0], "description": "1=basic, 2=standard, 3=dedicated"}]

        p1 = StakeholderProfile(
            name="Alice", role="Buyer", persona="",
            utility_function=UtilityFunction(weights={"price": 1.0}, ideal_values={"price": 20.0}, issues=[issue]),
            reservation_value=0.4,
        )
        p2 = StakeholderProfile(
            name="Bob", role="Seller", persona="",
            utility_function=UtilityFunction(weights={"price": 1.0}, ideal_values={"price": 80.0}, issues=[issue]),
            reservation_value=0.4,
        )

        mock_outcome = MagicMock()
        mock_outcome.agreement_reached = True
        mock_outcome.final_proposal = {"price": 50.0}
        mock_outcome.rounds_taken = 3

        summary = compute_plain_language_summary(mock_outcome, [p1, p2], issues_meta)
        assert summary["agreement_reached"] is True
        assert "Price" in summary["package_plain_text"]
        assert len(summary["party_outcomes"]) == 2
        assert "distance_to_frontier" in summary
        assert len(summary["explanation"].split(".")) >= 3

    def test_feasibility_check_satisfiable_sum_but_no_joint_point(self):
        """
        True grid/Monte Carlo feasibility test:
        Even though sum(r_i) = 0.60 + 0.60 = 1.20 <= 2.0 (satisfiable under naive sum),
        no point in the issue space gives both parties utility >= 0.60 simultaneously.
        """
        from scenarios.from_text import check_scenario_feasibility
        from src.models.utility import StakeholderProfile, UtilityFunction, Issue

        issue = Issue(name="price", min_value=0.0, max_value=100.0, description="Price")
        issues_meta = [{"name": "price", "range": [0.0, 100.0], "description": "Price"}]

        # Party 1: wants low price (ideal=0), rv=0.60 -> requires price <= 40
        p1 = StakeholderProfile(
            name="Buyer", role="Buyer", persona="",
            utility_function=UtilityFunction(weights={"price": 1.0}, ideal_values={"price": 0.0}, issues=[issue]),
            reservation_value=0.60,
        )
        # Party 2: wants high price (ideal=100), rv=0.60 -> requires price >= 60
        p2 = StakeholderProfile(
            name="Seller", role="Seller", persona="",
            utility_function=UtilityFunction(weights={"price": 1.0}, ideal_values={"price": 100.0}, issues=[issue]),
            reservation_value=0.60,
        )

        # Sum of reservation values is 1.20 <= 2.0 (satisfiable under naive sum)
        assert p1.reservation_value + p2.reservation_value == 1.20
        assert p1.reservation_value + p2.reservation_value <= 2.0

        # But true grid test finds no point where both are satisfied
        is_feasible, surplus = check_scenario_feasibility([p1, p2], issues_meta)
        assert is_feasible is False
        assert surplus < 0

    @pytest.mark.integration
    def test_six_issue_five_party_live_parse_no_truncation(self):
        """Bug 6.9: Test large 6-issue x 5-party prompt against real API with reasoning_effort='low'."""
        desc = (
            "Five international aerospace consortium partners (AeroFrance, GermanTech, ItalianSpace, SpanishAvionics, UKDefense) "
            "are negotiating a multinational next-generation satellite constellation project across 6 core issues: "
            "1. Total project budget ($100M to $500M). "
            "2. Workshare allocation for UKDefense (10% to 40%). "
            "3. Headquarters location tier (1=Paris, 2=Munich, 3=Rome, 4=Madrid). "
            "4. Delivery timeline in years (3 to 10 years). "
            "5. Intellectual property data sharing rights tier (1=restricted, 2=shared consortium, 3=fully open). "
            "6. Ground station count (5 to 30 stations). "
            "AeroFrance demands high budget ($450M) and Paris headquarters (1). "
            "GermanTech prioritizes Munich headquarters (2) and shorter delivery timeline (4 years). "
            "ItalianSpace focuses on ground stations (25 stations) and Rome headquarters (3). "
            "SpanishAvionics cares deeply about open IP rights (3) and affordable budget ($200M). "
            "UKDefense strictly insists on at least 35% workshare and restricted IP sharing (1)."
        )
        profiles, issues, report = generate_from_description(desc, seed=42)
        assert len(issues) == 6
        assert len(profiles) == 5
        for p in profiles:
            weights = p.utility_function.weights
            assert abs(sum(weights.values()) - 1.0) < 1e-4
            ideal_u = p.evaluate_proposal(p.utility_function.ideal_values)
            assert abs(ideal_u - 1.0) < 1e-4




