"""
Execute the README quickstart snippets with a mocked LLM.
Demonstrates:
1. Standard mediated negotiation quickstart snippet.
2. Strategic agents quickstart snippet with privacy measurement.
3. Robustness of privacy snippet when metrics are None (e.g., probe failure / LLM unavailable).
"""
import sys
import json
from unittest.mock import patch

from src.models.negotiation import (
    MediatorResponse,
    MediatorAction,
    Critique,
)

def run_quickstart_snippet_1():
    print("=" * 60)
    print("RUNNING QUICKSTART SNIPPET 1: Standard Mediated Negotiation")
    print("=" * 60)
    
    with patch("src.agents.mediator.structured_completion") as mock_med, \
         patch("src.agents.stakeholder.structured_completion") as mock_stk:
        
        # 1 mediator initial proposal, 1 agreement declaration
        mock_med.side_effect = [
            MediatorResponse(
                action=MediatorAction.PROPOSE,
                revised_proposal={
                    "unit_price": 55.0,
                    "order_volume": 5000.0,
                    "delivery_days": 15.0,
                    "payment_terms": 45.0,
                    "quality_tier": 3.0,
                },
                reasoning="Balanced initial compromise",
            ),
            MediatorResponse(
                action=MediatorAction.DECLARE_AGREEMENT,
                revised_proposal={
                    "unit_price": 55.0,
                    "order_volume": 5000.0,
                    "delivery_days": 15.0,
                    "payment_terms": 45.0,
                    "quality_tier": 3.0,
                },
                reasoning="All parties accept",
            ),
        ]
        
        # Stakeholders accept proposal
        mock_stk.return_value = Critique(
            agent_name="Stakeholder",
            satisfaction_score=8.0,
            acceptable=True,
            reasoning="Acceptable terms",
            round_number=1,
        )

        # Snippet code from README:
        from scenarios.business_deal import BusinessDealScenario
        from src.protocol.graph import run_negotiation

        scenario = BusinessDealScenario()
        profiles, issues = scenario.generate(seed=42)

        outcome = run_negotiation(
            profiles=profiles,
            issues=issues,
            scenario_name="business_deal",
            protocol="single_text",
            max_rounds=10,
        )

        print(f"Agreed: {outcome.agreement_reached}")
        print(f"Rounds: {outcome.rounds_taken}")
        print(f"Utilities: {outcome.per_agent_utilities}")
        print(f"Protocol: {outcome.protocol_used}")


def run_quickstart_snippet_2(simulate_none_metrics=False):
    print("\n" + "=" * 60)
    mode_str = "SIMULATING NONE METRICS (Probe Failure)" if simulate_none_metrics else "PROBE SUCCESS"
    print(f"RUNNING QUICKSTART SNIPPET 2: Strategic Agents & Privacy ({mode_str})")
    print("=" * 60)

    with patch("src.agents.mediator.structured_completion") as mock_med, \
         patch("src.agents.strategic_stakeholder.structured_completion") as mock_stk, \
         patch("src.eval.privacy.plain_completion") as mock_privacy:

        mock_med.side_effect = [
            MediatorResponse(
                action=MediatorAction.PROPOSE,
                revised_proposal={
                    "unit_price": 55.0,
                    "order_volume": 5000.0,
                    "delivery_days": 15.0,
                    "payment_terms": 45.0,
                    "quality_tier": 3.0,
                },
                reasoning="Balanced initial compromise",
            ),
            MediatorResponse(
                action=MediatorAction.DECLARE_AGREEMENT,
                revised_proposal={
                    "unit_price": 55.0,
                    "order_volume": 5000.0,
                    "delivery_days": 15.0,
                    "payment_terms": 45.0,
                    "quality_tier": 3.0,
                },
                reasoning="All parties accept",
            ),
        ]

        mock_stk.return_value = Critique(
            agent_name="StrategicAgent",
            satisfaction_score=7.5,
            acceptable=True,
            reasoning="Acceptable strategic concession",
            round_number=1,
        )

        if simulate_none_metrics:
            mock_privacy.side_effect = Exception("Probe LLM unavailable / network error")
        else:
            # Return inferred weights
            mock_privacy.return_value = json.dumps({
                "SupplierCo": {"unit_price": 0.4, "order_volume": 0.3, "delivery_days": 0.1, "payment_terms": 0.1, "quality_tier": 0.1},
                "BuyerInc": {"unit_price": 0.35, "order_volume": 0.25, "delivery_days": 0.15, "payment_terms": 0.15, "quality_tier": 0.1},
                "LogiTrans": {"unit_price": 0.1, "order_volume": 0.4, "delivery_days": 0.3, "payment_terms": 0.1, "quality_tier": 0.1},
            })

        # Snippet code from README:
        from scenarios.strategic_negotiation import StrategicNegotiationScenario
        from src.protocol.graph import run_negotiation
        from src.eval.privacy import measure_privacy_leakage

        scenario = StrategicNegotiationScenario()

        # Mixed-honesty population: honesty_level ~ Uniform(0.3, 1.0)
        profiles, issues = scenario.generate(seed=42)
        for p in profiles:
            print(f"  {p.name}: honesty_level={p.honesty_level:.2f} (strategic={p.is_strategic})")

        outcome = run_negotiation(
            profiles=profiles,
            issues=issues,
            scenario_name="strategic_negotiation",
        )

        # Measure how much the transcript leaked about each agent's preferences
        privacy = measure_privacy_leakage(
            transcript=outcome.messages,
            profiles=profiles,
            issues=issues,
        )
        cos_str = f"{privacy.mean_cosine_similarity:.3f}" if privacy.mean_cosine_similarity is not None else "N/A"
        u_str = f"{privacy.uniform_baseline_cosine:.3f}" if privacy.uniform_baseline_cosine is not None else "N/A"
        d_str = f"{privacy.dirichlet_baseline_cosine:.3f}" if privacy.dirichlet_baseline_cosine is not None else "N/A"
        lift_str = f"{privacy.lift_over_uniform:.3f}" if privacy.lift_over_uniform is not None else "N/A"
        print(f"Mean privacy leakage (cosine): {cos_str}")
        print(f"  Uniform-guess baseline: {u_str}")
        print(f"  Dirichlet baseline: {d_str}")
        print(f"  Lift over uniform: {lift_str}")
        print(f"Bluff suspects: {outcome.bluff_detection_scores}")


if __name__ == "__main__":
    run_quickstart_snippet_1()
    run_quickstart_snippet_2(simulate_none_metrics=False)
    run_quickstart_snippet_2(simulate_none_metrics=True)
