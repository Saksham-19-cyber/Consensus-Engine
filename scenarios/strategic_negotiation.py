"""
Strategic Negotiation Scenario
================================
A mixed-honesty 3-agent negotiation scenario derived from the business-deal structure.
Each agent is assigned an honesty_level drawn from Uniform(0.3, 1.0) per trial,
creating realistic heterogeneous populations of strategic and honest negotiators.

This scenario exists specifically to study the effects of strategic misrepresentation:
- How does population honesty_level affect Pareto efficiency and Nash welfare?
- Can the mediator's bluff detection reduce efficiency loss from strategic agents?
- Does the alternating-offers protocol handle bluffing differently from mediation?

Usage:
    scenario = StrategicNegotiationScenario()
    profiles, issues = scenario.generate(seed=42)
    # profiles[i].honesty_level < 1.0 → StrategicStakeholderAgent auto-selected
"""
from __future__ import annotations
import numpy as np
from scenarios.base import Scenario
from src.models.utility import Issue, UtilityFunction, StakeholderProfile


class StrategicNegotiationScenario(Scenario):
    name = "strategic_negotiation"
    description = (
        "Three-party negotiation with heterogeneous honesty levels. "
        "Each agent's honesty_level is drawn from Uniform(0.3, 1.0), "
        "creating mixed populations of bluffers and honest negotiators."
    )

    ISSUES = [
        Issue(name="unit_price", min_value=10.0, max_value=100.0, description="Price per unit in dollars"),
        Issue(name="order_volume", min_value=100.0, max_value=10000.0, description="Units per order"),
        Issue(name="delivery_days", min_value=1.0, max_value=30.0, description="Delivery time in days"),
        Issue(name="payment_terms", min_value=0.0, max_value=90.0, description="Net payment days"),
        Issue(name="quality_tier", min_value=1.0, max_value=5.0, description="Quality level 1-5"),
    ]

    def generate(self, seed: int = 42) -> tuple[list[StakeholderProfile], list[dict]]:
        rng = np.random.RandomState(seed)
        issue_names = [i.name for i in self.ISSUES]

        # Draw honesty levels: Uniform(0.3, 1.0) — at least partially strategic
        honesty_levels = rng.uniform(0.3, 1.0, size=3)

        def make_strategic_bias(
            ideal_values: dict[str, float],
            honesty_level: float,
            rng: np.random.RandomState,
            inflate_issues: list[str],
            inflate_direction: str = "higher",
        ) -> dict[str, float]:
            """
            Compute per-issue demand inflation offsets.
            Strategic agents inflate their stated ideal on high-priority issues
            by (1-honesty_level) * 20-40% of the issue range.
            """
            bias = {}
            for issue_name, issue in zip(issue_names, self.ISSUES):
                if issue_name in inflate_issues:
                    range_span = issue.max_value - issue.min_value
                    magnitude = (1.0 - honesty_level) * rng.uniform(0.2, 0.4) * range_span
                    bias[issue_name] = magnitude if inflate_direction == "higher" else -magnitude
                else:
                    bias[issue_name] = 0.0
            return bias

        # --- Supplier (wants high price, large volume, long delivery, short payment) ---
        supplier_honesty = honesty_levels[0]
        supplier_weights = rng.dirichlet([4, 2, 1, 3, 1])
        supplier_ideals = {
            "unit_price": 70.0 + rng.uniform(0, 30),
            "order_volume": 5000.0 + rng.uniform(0, 5000),
            "delivery_days": 15.0 + rng.uniform(0, 15),
            "payment_terms": 10.0 + rng.uniform(0, 20),
            "quality_tier": 3.0 + rng.uniform(0, 1),
        }
        supplier_bias = make_strategic_bias(
            supplier_ideals, supplier_honesty, rng,
            inflate_issues=["unit_price", "payment_terms"],
            inflate_direction="higher",
        )
        supplier = StakeholderProfile(
            name="SupplierCo",
            role="Supplier",
            persona="Maximizes revenue; strategic on price and payment terms",
            utility_function=UtilityFunction(
                weights=dict(zip(issue_names, supplier_weights.tolist())),
                ideal_values=supplier_ideals,
                issues=self.ISSUES,
            ),
            reservation_value=round(0.3 + rng.uniform(0, 0.1), 2),
            honesty_level=round(float(supplier_honesty), 3),
            strategic_bias=supplier_bias,
        )

        # --- Buyer (wants low price, fast delivery, high quality) ---
        buyer_honesty = honesty_levels[1]
        buyer_weights = rng.dirichlet([4, 2, 3, 1, 3])
        buyer_ideals = {
            "unit_price": 10.0 + rng.uniform(0, 20),
            "order_volume": 500.0 + rng.uniform(0, 2000),
            "delivery_days": 1.0 + rng.uniform(0, 5),
            "payment_terms": 60.0 + rng.uniform(0, 30),
            "quality_tier": 4.0 + rng.uniform(0, 1),
        }
        buyer_bias = make_strategic_bias(
            buyer_ideals, buyer_honesty, rng,
            inflate_issues=["delivery_days", "quality_tier"],
            inflate_direction="lower",   # buyer inflates demand downward (wants faster/better)
        )
        buyer = StakeholderProfile(
            name="BuyerInc",
            role="Buyer",
            persona="Minimizes cost; strategic on delivery time and quality demands",
            utility_function=UtilityFunction(
                weights=dict(zip(issue_names, buyer_weights.tolist())),
                ideal_values=buyer_ideals,
                issues=self.ISSUES,
            ),
            reservation_value=round(0.3 + rng.uniform(0, 0.1), 2),
            honesty_level=round(float(buyer_honesty), 3),
            strategic_bias=buyer_bias,
        )

        # --- Logistics (cares about volume and delivery window) ---
        logistics_honesty = honesty_levels[2]
        logistics_weights = rng.dirichlet([1, 3, 4, 1, 1])
        logistics_ideals = {
            "unit_price": 55.0,
            "order_volume": 2000.0 + rng.uniform(0, 3000),
            "delivery_days": 7.0 + rng.uniform(0, 7),
            "payment_terms": 30.0,
            "quality_tier": 3.0,
        }
        logistics_bias = make_strategic_bias(
            logistics_ideals, logistics_honesty, rng,
            inflate_issues=["delivery_days", "order_volume"],
            inflate_direction="higher",  # logistics wants more delivery time buffer
        )
        logistics = StakeholderProfile(
            name="LogiTrans",
            role="Logistics Provider",
            persona="Optimizes for realistic delivery windows; strategic on volume and timing",
            utility_function=UtilityFunction(
                weights=dict(zip(issue_names, logistics_weights.tolist())),
                ideal_values=logistics_ideals,
                issues=self.ISSUES,
            ),
            reservation_value=round(0.3 + rng.uniform(0, 0.1), 2),
            honesty_level=round(float(logistics_honesty), 3),
            strategic_bias=logistics_bias,
        )

        return [supplier, buyer, logistics], self.get_issues_meta(self.ISSUES)

    def generate_honest(self, seed: int = 42) -> tuple[list[StakeholderProfile], list[dict]]:
        """
        Generate an all-honest population (honesty_level=1.0 for all agents).
        Useful for ablation: compare against mixed-honesty to isolate the
        effect of strategic misrepresentation.
        """
        profiles, issues = self.generate(seed=seed)
        honest_profiles = []
        for p in profiles:
            honest_profiles.append(p.model_copy(update={"honesty_level": 1.0, "strategic_bias": {}}))
        return honest_profiles, issues

    def generate_all_strategic(self, seed: int = 42, honesty: float = 0.3) -> tuple[list[StakeholderProfile], list[dict]]:
        """
        Generate an all-strategic population with fixed honesty_level.
        Useful for ablation: full-bluff vs. mixed vs. all-honest comparison.
        """
        profiles, issues = self.generate(seed=seed)
        strategic_profiles = []
        for p in profiles:
            strategic_profiles.append(p.model_copy(update={"honesty_level": honesty}))
        return strategic_profiles, issues
