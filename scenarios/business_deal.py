from __future__ import annotations
import numpy as np
from scenarios.base import Scenario
from src.models.utility import Issue, UtilityFunction, StakeholderProfile


class BusinessDealScenario(Scenario):
    name = "business_deal"
    description = "Three-party negotiation: supplier, buyer, and logistics provider"

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

        supplier_weights = rng.dirichlet([4, 2, 1, 3, 1])
        supplier = StakeholderProfile(
            name="SupplierCo",
            role="Supplier",
            persona="Maximizes revenue, prefers large orders with longer payment and delivery windows, flexible on quality",
            utility_function=UtilityFunction(
                weights=dict(zip(issue_names, supplier_weights.tolist())),
                ideal_values={
                    "unit_price": 70.0 + rng.uniform(0, 30),
                    "order_volume": 5000.0 + rng.uniform(0, 5000),
                    "delivery_days": 15.0 + rng.uniform(0, 15),
                    "payment_terms": 10.0 + rng.uniform(0, 20),
                    "quality_tier": 3.0 + rng.uniform(0, 1),
                },
                issues=self.ISSUES,
            ),
            reservation_value=round(0.3 + rng.uniform(0, 0.1), 2),
        )

        buyer_weights = rng.dirichlet([4, 2, 3, 1, 3])
        buyer = StakeholderProfile(
            name="BuyerInc",
            role="Buyer",
            persona="Minimizes cost, needs fast delivery and high quality, flexible on volume and payment terms",
            utility_function=UtilityFunction(
                weights=dict(zip(issue_names, buyer_weights.tolist())),
                ideal_values={
                    "unit_price": 10.0 + rng.uniform(0, 20),
                    "order_volume": 500.0 + rng.uniform(0, 2000),
                    "delivery_days": 1.0 + rng.uniform(0, 5),
                    "payment_terms": 60.0 + rng.uniform(0, 30),
                    "quality_tier": 4.0 + rng.uniform(0, 1),
                },
                issues=self.ISSUES,
            ),
            reservation_value=round(0.3 + rng.uniform(0, 0.1), 2),
        )

        logistics_weights = rng.dirichlet([1, 3, 4, 1, 1])
        logistics = StakeholderProfile(
            name="LogiTrans",
            role="Logistics Provider",
            persona="Optimizes for manageable volume and realistic delivery times, indifferent to price and payment",
            utility_function=UtilityFunction(
                weights=dict(zip(issue_names, logistics_weights.tolist())),
                ideal_values={
                    "unit_price": 55.0,
                    "order_volume": 2000.0 + rng.uniform(0, 3000),
                    "delivery_days": 7.0 + rng.uniform(0, 7),
                    "payment_terms": 30.0,
                    "quality_tier": 3.0,
                },
                issues=self.ISSUES,
            ),
            reservation_value=round(0.3 + rng.uniform(0, 0.1), 2),
        )

        return [supplier, buyer, logistics], self.get_issues_meta(self.ISSUES)
