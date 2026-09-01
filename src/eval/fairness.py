from __future__ import annotations
import numpy as np


def nash_social_welfare(utilities: np.ndarray) -> float:
    clipped = np.maximum(utilities, 1e-10)
    return float(np.exp(np.sum(np.log(clipped))))


def utilitarian_welfare(utilities: np.ndarray) -> float:
    return float(np.sum(utilities))


def egalitarian_welfare(utilities: np.ndarray) -> float:
    return float(np.min(utilities))


def gini_coefficient(utilities: np.ndarray) -> float:
    n = len(utilities)
    if n == 0:
        return 0.0
    sorted_u = np.sort(utilities)
    mean_u = np.mean(sorted_u)
    if mean_u == 0:
        return 0.0
    numerator = 0.0
    for i in range(n):
        for j in range(n):
            numerator += abs(sorted_u[i] - sorted_u[j])
    return float(numerator / (2 * n * n * mean_u))


def envy_freeness_check(utilities: dict[str, float]) -> bool:
    values = list(utilities.values())
    max_val = max(values) if values else 0
    min_val = min(values) if values else 0
    return (max_val - min_val) < 0.1


def compute_fairness_metrics(utilities: dict[str, float]) -> dict:
    u_arr = np.array(list(utilities.values()))
    return {
        "nash_welfare": nash_social_welfare(u_arr),
        "min_utility": float(np.min(u_arr)) if len(u_arr) > 0 else 0.0,
        "max_utility": float(np.max(u_arr)) if len(u_arr) > 0 else 0.0,
        "gini_coefficient": gini_coefficient(u_arr),
        "envy_free": envy_freeness_check(utilities),
        "utility_spread": float(np.max(u_arr) - np.min(u_arr)) if len(u_arr) > 0 else 0.0,
    }
