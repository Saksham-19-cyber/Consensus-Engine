from __future__ import annotations
import json
from src.models.evaluation import EvalReport, TrialResult


def generate_markdown_report(
    scenario_name: str,
    summary: dict[str, dict[str, float]],
    results: list[TrialResult],
) -> str:
    lines = [
        f"# Evaluation Report: {scenario_name}",
        f"\nTotal trials: {len(results)}",
        "",
        "## Summary by Method",
        "",
        "| Method | Agree% | Pareto Ratio | Nash Welfare | Min Utility | Gini | Trials |",
        "|--------|--------|-------------|-------------|-------------|------|--------|",
    ]

    for method, stats in sorted(summary.items()):
        lines.append(
            f"| {method} "
            f"| {stats['agreement_rate']:.1%} "
            f"| {stats['mean_pareto_ratio']:.3f}±{stats['std_pareto_ratio']:.3f} "
            f"| {stats['mean_nash_welfare']:.3f}±{stats['std_nash_welfare']:.3f} "
            f"| {stats['mean_min_utility']:.3f} "
            f"| {stats['mean_gini']:.3f} "
            f"| {int(stats['n_trials'])} |"
        )

    lines.extend(["", "## Key Findings", ""])

    methods_sorted = sorted(summary.items(), key=lambda x: x[1]["mean_nash_welfare"], reverse=True)
    if methods_sorted:
        best = methods_sorted[0]
        lines.append(f"- **Best Nash Welfare**: {best[0]} ({best[1]['mean_nash_welfare']:.3f})")

    methods_sorted_pareto = sorted(summary.items(), key=lambda x: x[1]["mean_pareto_ratio"], reverse=True)
    if methods_sorted_pareto:
        best_p = methods_sorted_pareto[0]
        lines.append(f"- **Best Pareto Ratio**: {best_p[0]} ({best_p[1]['mean_pareto_ratio']:.3f})")

    methods_sorted_fair = sorted(summary.items(), key=lambda x: x[1]["mean_gini"])
    if methods_sorted_fair:
        best_f = methods_sorted_fair[0]
        lines.append(f"- **Most Fair (lowest Gini)**: {best_f[0]} ({best_f[1]['mean_gini']:.3f})")

    return "\n".join(lines)


def generate_json_report(
    scenario_name: str,
    summary: dict[str, dict[str, float]],
    results: list[TrialResult],
) -> str:
    report = EvalReport(
        scenario_name=scenario_name,
        n_trials=len(results),
        methods=list(summary.keys()),
        results=results,
        summary=summary,
    )
    return report.model_dump_json(indent=2)
