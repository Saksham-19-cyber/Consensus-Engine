from __future__ import annotations
import numpy as np
from itertools import product
from src.models.utility import UtilityFunction


def is_dominated(u: np.ndarray, candidates: np.ndarray) -> bool:
    for c in candidates:
        if np.all(c >= u) and np.any(c > u):
            return True
    return False


def compute_pareto_frontier(
    utility_functions: list[UtilityFunction],
    issues: list[dict],
    resolution: int = 12,
) -> np.ndarray:
    issue_names = [i["name"] for i in issues]
    grids = [np.linspace(i["range"][0], i["range"][1], resolution) for i in issues]
    mesh = np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(issues), -1).T

    n_combos = mesh.shape[0]
    all_utilities = np.zeros((n_combos, len(utility_functions)))
    issue_map = {i["name"]: i for i in issues}

    for p_idx, uf in enumerate(utility_functions):
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
        all_utilities[:, p_idx] = np.clip(score_acc, 0.0, 1.0)

    unique_utilities = np.unique(np.round(all_utilities, 4), axis=0)

    if len(unique_utilities) == 0:
        return np.array([])

    frontier_mask = np.ones(len(unique_utilities), dtype=bool)
    for i in range(len(unique_utilities)):
        if not frontier_mask[i]:
            continue
        u_i = unique_utilities[i]
        for j in range(len(unique_utilities)):
            if i == j or not frontier_mask[j]:
                continue
            u_j = unique_utilities[j]
            if np.all(u_j >= u_i) and np.any(u_j > u_i):
                frontier_mask[i] = False
                break

    return unique_utilities[frontier_mask]


def pareto_distance(
    outcome_utilities: np.ndarray, frontier: np.ndarray
) -> float:
    if len(frontier) == 0:
        return float("inf")
    distances = np.linalg.norm(frontier - outcome_utilities, axis=1)
    return float(np.min(distances))


def pareto_efficiency_ratio(
    outcome_utilities: np.ndarray, frontier: np.ndarray
) -> float:
    if len(frontier) == 0:
        return 0.0
    outcome_welfare = float(np.sum(outcome_utilities))
    frontier_welfares = np.sum(frontier, axis=1)
    max_welfare = float(np.max(frontier_welfares))
    if max_welfare == 0:
        return 0.0
    return min(1.0, outcome_welfare / max_welfare)


def find_optimal_proposal(
    utility_functions: list[UtilityFunction],
    issues: list[dict],
    resolution: int = 20,
    metric: str = "nash",
) -> tuple[dict[str, float], np.ndarray]:
    issue_names = [i["name"] for i in issues]
    grids = [np.linspace(i["range"][0], i["range"][1], resolution) for i in issues]

    best_score = -float("inf")
    best_proposal = {}
    best_utilities = np.array([])

    for combo in product(*grids):
        proposal = dict(zip(issue_names, combo))
        utilities = np.array([uf.score(proposal) for uf in utility_functions])

        if metric == "nash":
            score = float(np.sum(np.log(np.maximum(utilities, 1e-10))))
        elif metric == "utilitarian":
            score = float(np.sum(utilities))
        elif metric == "egalitarian":
            score = float(np.min(utilities))
        else:
            score = float(np.sum(utilities))

        if score > best_score:
            best_score = score
            best_proposal = proposal
            best_utilities = utilities

    return best_proposal, best_utilities
