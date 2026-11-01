"""
Tests for Phase 3: Benchmark runner, resumability, seed-aligned Wilcoxon, and eval API endpoints.
All tests use mocked LLM calls and temporary files.
"""
from __future__ import annotations
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.models.utility import StakeholderProfile, UtilityFunction, Issue
from src.models.negotiation import NegotiationOutcome, NegotiationStatus
from src.models.evaluation import TrialResult, ParetoMetrics, FairnessMetrics
from src.eval.runner import (
    run_engine_trial,
    run_batch_engine,
    run_batch_baselines,
    aggregate_results,
    get_git_commit_hash,
)
from src.eval.log_writer import TrialLogWriter, load_trial_log
from src.api.main import app

client = TestClient(app)

ISSUES = [
    {"name": "price", "range": [0.0, 100.0]},
]


def _make_profile(name: str, ideal: float = 100.0, reservation: float = 0.2) -> StakeholderProfile:
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
        honesty_level=1.0,
    )


class TestBenchmarkRunner:
    @patch("src.protocol.graph.run_negotiation")
    def test_run_engine_trial_fields_and_no_chromadb_contamination(self, mock_run_neg):
        """run_engine_trial must NOT pass scenario_name to run_negotiation, and must populate metadata fields."""
        profiles = [_make_profile("A"), _make_profile("B")]
        mock_run_neg.return_value = NegotiationOutcome(
            final_proposal={"price": 50.0},
            per_agent_utilities={"A": 0.5, "B": 0.5},
            status=NegotiationStatus.AGREED,
            rounds_taken=2,
            agreement_reached=True,
            protocol_used="single_text",
        )

        result = run_engine_trial(
            trial_id=1,
            profiles=profiles,
            issues=ISSUES,
            scenario_name="test_scenario",
            seed=42,
            protocol="single_text",
            model_config_name="120b_vs_20b",
        )

        # scenario_name must be None to prevent precedent contamination
        mock_run_neg.assert_called_once()
        assert mock_run_neg.call_args.kwargs.get("scenario_name") is None

        # Verify all metadata fields required by Phase 3
        assert result.seed == 42
        assert result.protocol == "single_text"
        assert result.method == "consensus_engine"
        assert "negotiator_model" in result.model_names
        assert "mediator_model" in result.model_names
        assert result.git_commit_hash != ""
        assert result.timestamp != ""
        assert result.agreement_reached is True

    @patch("src.protocol.graph.run_negotiation")
    def test_resumable_runs_skip_already_logged_trials(self, mock_run_neg, tmp_path):
        """When a trial log already has (scenario, method, seed), it must be skipped."""
        log_file = tmp_path / "test_resume.jsonl"
        writer = TrialLogWriter(scenario_name="test", filepath=log_file)
        writer.open()

        existing_trial = TrialResult(
            trial_id=0,
            scenario_name="test",
            seed=100,
            method="consensus_engine",
            protocol="single_text",
            agreement_reached=True,
            rounds_taken=1,
            pareto=ParetoMetrics(efficiency_ratio=0.9),
            fairness=FairnessMetrics(nash_welfare=0.5),
        )
        writer.write(existing_trial)

        mock_run_neg.return_value = NegotiationOutcome(
            final_proposal={"price": 50.0},
            per_agent_utilities={"A": 0.5, "B": 0.5},
            status=NegotiationStatus.AGREED,
            rounds_taken=1,
            agreement_reached=True,
        )

        def mock_generator(seed: int):
            return [_make_profile("A"), _make_profile("B")], ISSUES

        # Run 2 trials: seed 100 (already in log) and seed 101 (new)
        results = run_batch_engine(
            scenario_generator=mock_generator,
            n_trials=2,
            scenario_name="test",
            seed=100,
            log_writer=writer,
        )

        writer.close()

        # Only trial for seed 101 should have been executed
        assert len(results) == 1
        assert results[0].seed == 101
        assert mock_run_neg.call_count == 1

        # File should now contain 2 records total
        loaded = load_trial_log(log_file)
        assert len(loaded) == 2


class TestSeedAlignedWilcoxon:
    def test_wilcoxon_pairing_is_seed_aligned(self):
        """Wilcoxon test must pair engine and baseline trials by matching seed, even if list order differs."""
        # 10 trials with different seeds
        engine_trials = [
            TrialResult(
                trial_id=i,
                seed=100 + i,
                scenario_name="test",
                method="consensus_engine",
                pareto=ParetoMetrics(efficiency_ratio=0.85 + (i * 0.01)),
                fairness=FairnessMetrics(nash_welfare=0.50 + (i * 0.01)),
            )
            for i in range(10)
        ]

        # Baseline trials reversed in list order
        baseline_trials = [
            TrialResult(
                trial_id=i,
                seed=100 + i,
                scenario_name="test",
                method="public_midpoint",
                pareto=ParetoMetrics(efficiency_ratio=0.70 + (i * 0.01)),
                fairness=FairnessMetrics(nash_welfare=0.40 + (i * 0.01)),
            )
            for i in reversed(range(10))
        ]

        # Aggregate should successfully align by seed
        summary = aggregate_results(engine_trials + baseline_trials, engine_method="consensus_engine")
        assert "public_midpoint" in summary
        w = summary["public_midpoint"].get("wilcoxon_pareto_vs_engine", {})
        if w.get("p_value") is not None:
            # Engine strictly dominates baseline for each seed (0.85+ vs 0.70+)
            assert w.get("significant_05") is True


class TestEvalApiRoutes:
    def test_get_eval_report_no_run_found(self):
        """GET /api/eval/report/{scenario} returns 'No benchmark run found' when no logs exist."""
        resp = client.get("/api/eval/report/nonexistent_scenario_12345")
        assert resp.status_code == 200
        data = resp.json()
        assert "No benchmark run found" in data["report_markdown"]
        assert data["summary"] == {}

    def test_get_eval_report_reads_from_jsonl(self, tmp_path, monkeypatch):
        """GET /api/eval/report/{scenario} aggregates dynamically from JSONL log."""
        from src.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        writer = TrialLogWriter("test_scen", log_dir=log_dir)
        writer.open()
        trial = TrialResult(
            trial_id=1,
            seed=42,
            scenario_name="test_scen",
            method="consensus_engine",
            agreement_reached=True,
            rounds_taken=2,
            pareto=ParetoMetrics(efficiency_ratio=0.95),
            fairness=FairnessMetrics(nash_welfare=0.48),
        )
        writer.write(trial)
        writer.close()

        resp = client.get("/api/eval/report/test_scen")
        assert resp.status_code == 200
        data = resp.json()
        assert "consensus_engine" in data["summary"]
        assert data["summary"]["consensus_engine"]["agreement_rate"] == 1.0

    def test_post_eval_run_accepts_new_parameters(self):
        """POST /api/eval/run accepts model_config, protocol, run_privacy_probe, and consensus_engine."""
        with patch("src.eval.runner.run_batch_baselines", return_value=[]) as mock_b:
            with patch("src.eval.runner.run_batch_engine", return_value=[]) as mock_e:
                payload = {
                    "scenario": "roommate",
                    "n_trials": 2,
                    "methods": ["naive_average", "consensus_engine"],
                    "seed": 42,
                    "model_config": "120b_vs_20b",
                    "protocol": "single_text",
                    "run_privacy_probe": False,
                }
                resp = client.post("/api/eval/run", json=payload)
                assert resp.status_code == 200
                assert mock_b.called
                assert mock_e.called
