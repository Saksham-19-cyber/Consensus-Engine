from __future__ import annotations
import json
import numpy as np
from src.models.utility import StakeholderProfile, UtilityFunction
from src.eval.pareto import find_optimal_proposal
from src.llm.client import structured_completion
from src.config import settings
from pydantic import BaseModel, Field


class OracleProposal(BaseModel):
    proposal: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""


def naive_average_baseline(
    profiles: list[StakeholderProfile],
    issues: list[dict],
) -> dict[str, float]:
    result = {}
    for issue in issues:
        name = issue["name"]
        ideals = []
        for p in profiles:
            if name in p.utility_function.ideal_values:
                ideals.append(p.utility_function.ideal_values[name])
            else:
                ideals.append((issue["range"][0] + issue["range"][1]) / 2)
        result[name] = float(np.mean(ideals))
    return result


def nash_bargaining_baseline(
    profiles: list[StakeholderProfile],
    issues: list[dict],
    resolution: int = 12,
) -> dict[str, float]:
    issue_names = [i["name"] for i in issues]
    grids = [np.linspace(i["range"][0], i["range"][1], resolution) for i in issues]
    mesh = np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(issues), -1).T

    reservation_values = np.array([p.reservation_value for p in profiles])
    n_combos = mesh.shape[0]
    all_utilities = np.zeros((len(profiles), n_combos))

    for p_idx, p in enumerate(profiles):
        uf = p.utility_function
        issue_map = {i["name"]: i for i in issues}
        score_acc = np.zeros(n_combos)
        for issue_idx, issue_name in enumerate(issue_names):
            weight = uf.weights.get(issue_name, 0.0)
            if weight == 0:
                continue
            issue_meta = issue_map[issue_name]
            min_v, max_v = issue_meta["range"]
            span = max_v - min_v if max_v != min_v else 1.0
            ideal = uf.ideal_values.get(issue_name, (max_v + min_v) / 2)

            vals = mesh[:, issue_idx]
            norm_val = (vals - min_v) / span
            norm_ideal = (ideal - min_v) / span
            distance = np.abs(norm_val - norm_ideal)
            score_acc += weight * (1.0 - distance)
        all_utilities[p_idx] = np.clip(score_acc, 0.0, 1.0)

    surplus = all_utilities - reservation_values[:, None]
    valid_mask = np.all(surplus >= 0, axis=0)

    if not np.any(valid_mask):
        return {i["name"]: (i["range"][0] + i["range"][1]) / 2 for i in issues}

    valid_surplus = surplus[:, valid_mask]
    scores = np.sum(np.log(np.maximum(valid_surplus, 1e-10)), axis=0)

    valid_indices = np.where(valid_mask)[0]
    best_combo_idx = valid_indices[np.argmax(scores)]
    best_vals = mesh[best_combo_idx]

    return {name: round(float(best_vals[idx]), 4) for idx, name in enumerate(issue_names)}


def single_llm_oracle_baseline(
    profiles: list[StakeholderProfile],
    issues: list[dict],
) -> dict[str, float]:
    all_info = []
    for p in profiles:
        all_info.append({
            "name": p.name,
            "role": p.role,
            "weights": p.utility_function.weights,
            "ideal_values": p.utility_function.ideal_values,
            "reservation_value": p.reservation_value,
        })

    issues_desc = json.dumps(issues, indent=2)
    agents_desc = json.dumps(all_info, indent=2)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an optimization expert. Given full knowledge of all parties' "
                "utility functions and reservation values, find the proposal that "
                "maximizes the Nash social welfare (product of utilities). "
                "All parties must get utility >= their reservation value."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Issues:\n{issues_desc}\n\n"
                f"Agents:\n{agents_desc}\n\n"
                "Find the optimal proposal."
            ),
        },
    ]

    try:
        result = structured_completion(
            messages=messages,
            response_model=OracleProposal,
            model=settings.negotiator_model,
            temperature=0.2,
        )
        return result.proposal
    except Exception:
        return naive_average_baseline(profiles, issues)


def compute_baseline(
    method: str,
    profiles: list[StakeholderProfile],
    issues: list[dict],
) -> dict[str, float]:
    if method == "naive_average":
        return naive_average_baseline(profiles, issues)
    elif method == "nash_bargaining":
        return nash_bargaining_baseline(profiles, issues)
    elif method == "single_llm_oracle":
        return single_llm_oracle_baseline(profiles, issues)
    else:
        raise ValueError(f"Unknown baseline method: {method}")
