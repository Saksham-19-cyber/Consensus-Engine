"""
Unit tests for Custom Scenario API flows:
- Bug 6.5: protocol and model_config pass-through.
- Bug 6.6: FastAPI exception handlers returning {detail, error_type} with CORS headers.
- Bug 6.8: Background job execution and polling via GET /api/negotiate/{job_id}.
- Bug 6.3: User edit update endpoint POST /api/scenario/update.
"""
from __future__ import annotations
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest
from groq import RateLimitError, APIConnectionError

from src.api.main import app
from scenarios.from_text import ParseReport
from src.models.utility import StakeholderProfile, UtilityFunction, Issue
from src.models.negotiation import NegotiationOutcome, NegotiationStatus, NegotiationMessage


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def _make_dummy_token():
    issue = Issue(name="price", min_value=10.0, max_value=100.0, description="Price in dollars")
    p1 = StakeholderProfile(
        name="Buyer",
        role="Buyer",
        persona="Wants low price",
        utility_function=UtilityFunction(weights={"price": 1.0}, ideal_values={"price": 10.0}, issues=[issue]),
        reservation_value=0.3,
    )
    p2 = StakeholderProfile(
        name="Seller",
        role="Seller",
        persona="Wants high price",
        utility_function=UtilityFunction(weights={"price": 1.0}, ideal_values={"price": 100.0}, issues=[issue]),
        reservation_value=0.3,
    )
    report = ParseReport(
        issues=[{"name": "price", "range": [10.0, 100.0], "description": "Price in dollars"}],
        profiles=[p1.model_dump(), p2.model_dump()],
        stakeholder_sources={"Buyer": "user_specified", "Seller": "llm_inferred"},
        is_feasible=True,
    )
    return report.to_token()


class TestCustomScenarioAPI:
    def test_error_handlers_and_cors_headers(self, client):
        """Bug 6.6: RateLimitError, APIConnectionError, and validation errors return proper status, JSON body, and CORS headers."""
        # 1. Validation error (body too short) -> 422
        res_val = client.post("/api/scenario/parse", json={"description": "short"})
        assert res_val.status_code == 422
        data_val = res_val.json()
        assert "error_type" in data_val or "detail" in data_val
        assert res_val.headers.get("access-control-allow-origin") == "*"

        # 2. Mock RateLimitError on parse
        with patch("scenarios.from_text.structured_completion") as mock_sc:
            mock_sc.side_effect = RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429, headers={}),
                body={"error": "Rate limit"},
            )
            res_rate = client.post("/api/scenario/parse", json={"description": "A valid negotiation text that is long enough."})
            assert res_rate.status_code == 429
            data_rate = res_rate.json()
            assert data_rate.get("error_type") == "rate_limit"
            assert res_rate.headers.get("access-control-allow-origin") == "*"

        # 3. Mock APIConnectionError on parse
        with patch("scenarios.from_text.structured_completion") as mock_sc:
            mock_sc.side_effect = APIConnectionError(request=MagicMock())
            res_conn = client.post("/api/scenario/parse", json={"description": "A valid negotiation text that is long enough."})
            assert res_conn.status_code == 502
            data_conn = res_conn.json()
            assert data_conn.get("error_type") == "upstream_error"
            assert res_conn.headers.get("access-control-allow-origin") == "*"

    @patch("src.api.routes.run_negotiation")
    def test_protocol_and_model_config_passthrough_and_background_polling(self, mock_run, client):
        """Bug 6.5 & 6.8: Verify protocol, model_config are passed through, and job polling returns progress and summary."""
        token = _make_dummy_token()
        captured_kwargs = {}

        def fake_run(**kwargs):
            captured_kwargs.update(kwargs)
            on_msg = kwargs.get("on_message")
            if on_msg:
                on_msg(NegotiationMessage(
                    role="mediator",
                    agent_name="Mediator",
                    content="Proposed compromise",
                    round_number=1,
                    message_type="proposal",
                    metadata={"proposal": {"price": 55.0}},
                ))
            return NegotiationOutcome(
                status=NegotiationStatus.AGREED,
                agreement_reached=True,
                final_proposal={"price": 55.0},
                rounds_taken=1,
                messages=[],
            )

        mock_run.side_effect = fake_run

        # Start negotiation (background)
        res_start = client.post(
            "/api/negotiate",
            json={
                "parsed_scenario_token": token,
                "protocol": "alternating_offers",
                "model_config": "120b_vs_20b",
                "max_rounds": 4,
            },
        )
        assert res_start.status_code == 200
        start_data = res_start.json()
        assert "job_id" in start_data
        job_id = start_data["job_id"]

        # Poll job status
        res_poll = client.get(f"/api/negotiate/{job_id}")
        assert res_poll.status_code == 200
        poll_data = res_poll.json()
        assert poll_data["job_id"] == job_id
        # Wait until job completed
        import time
        for _ in range(20):
            if poll_data["status"] == "completed":
                break
            time.sleep(0.1)
            res_poll = client.get(f"/api/negotiate/{job_id}")
            poll_data = res_poll.json()

        assert poll_data["status"] == "completed"
        assert poll_data["summary"] is not None
        assert "package_plain_text" in poll_data["summary"]
        assert captured_kwargs.get("protocol") == "alternating_offers"
        assert captured_kwargs.get("model_config") == "120b_vs_20b"

    def test_update_scenario_endpoint(self, client):
        """Bug 6.3: POST /api/scenario/update accepts user edits and returns re-validated report with token."""
        edit_payload = {
            "issues": [
                {"name": "salary", "range": [60000.0, 120000.0], "description": "USD salary"},
                {"name": "remote_days", "range": [0.0, 5.0], "description": "Days per week"},
            ],
            "stakeholders": [
                {
                    "name": "Candidate",
                    "role": "Candidate",
                    "persona": "Engineer",
                    "weights": {"salary": 0.8, "remote_days": 0.2},
                    "ideal_values": {"salary": 110000.0, "remote_days": 4.0},
                    "reservation_value": 0.5,
                },
                {
                    "name": "Company",
                    "role": "Company",
                    "persona": "Employer",
                    "weights": {"salary": 0.6, "remote_days": 0.4},
                    "ideal_values": {"salary": 80000.0, "remote_days": 2.0},
                    "reservation_value": 0.4,
                },
            ],
        }

        res = client.post("/api/scenario/update", json=edit_payload)
        assert res.status_code == 200
        data = res.json()
        assert len(data["issues"]) == 2
        assert len(data["stakeholders"]) == 2
        assert data["stakeholders"][0]["source"] == "user_specified"
        assert data["is_feasible"] is True


def test_endpoint_caps_and_rate_limiting(client):
    import time
    from src.api.routes import _IP_REQUEST_LOG, _RATE_LIMIT_MAX_REQUESTS
    _IP_REQUEST_LOG.clear()

    token = _make_dummy_token()

    # Test max_rounds cap (> 10)
    res_rounds = client.post(
        "/api/negotiate/start",
        json={"parsed_scenario_token": token, "max_rounds": 15},
    )
    assert res_rounds.status_code == 400
    assert "max_rounds exceeds maximum" in res_rounds.json()["detail"]

    # Test n_trials cap (> 5)
    res_trials = client.post(
        "/api/eval/run",
        json={"scenario": "roommate", "n_trials": 6},
    )
    assert res_trials.status_code == 400
    assert "n_trials exceeds maximum" in res_trials.json()["detail"]

    # Test client IP resolution from X-Forwarded-For and X-Real-IP headers
    from src.api.routes import get_client_ip
    from starlette.requests import Request

    scope_xff = {
        "type": "http",
        "headers": [(b"x-forwarded-for", b"203.0.113.195, 70.41.3.18"), (b"host", b"localhost")],
        "client": ("127.0.0.1", 12345),
    }
    req_xff = Request(scope_xff)
    assert get_client_ip(req_xff) == "203.0.113.195"

    scope_real = {
        "type": "http",
        "headers": [(b"x-real-ip", b"198.51.100.1"), (b"host", b"localhost")],
        "client": ("127.0.0.1", 12345),
    }
    req_real = Request(scope_real)
    assert get_client_ip(req_real) == "198.51.100.1"

    scope_direct = {
        "type": "http",
        "headers": [(b"host", b"localhost")],
        "client": ("192.0.2.1", 12345),
    }
    req_direct = Request(scope_direct)
    assert get_client_ip(req_direct) == "192.0.2.1"

    # Test rate limit trigger by populating the window with 10 timestamps
    now = time.time()
    _IP_REQUEST_LOG["testclient"] = [now] * _RATE_LIMIT_MAX_REQUESTS

    # 11th call should trigger 429
    with patch("src.api.routes._execute_negotiation_job"):
        res_429 = client.post(
            "/api/negotiate/start",
            json={"parsed_scenario_token": token, "max_rounds": 5},
        )
    assert res_429.status_code == 429
    assert "Rate limit exceeded" in res_429.json()["detail"]

    # Clean up state
    _IP_REQUEST_LOG.clear()
