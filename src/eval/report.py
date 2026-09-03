"""
Evaluation Report Generator
=============================
Generates Markdown and JSON evaluation reports with:
  - mean ± 95% CI per metric per method
  - Significance markers (*, **, n.s.) from Wilcoxon tests vs. Consensus Engine
  - Privacy leakage summary when available
  - Bluff detection statistics when available
"""
from __future__ import annotations
import json
from src.models.evaluation import EvalReport, TrialResult


def _sig_marker(wilcoxon_entry: dict | None) -> str:
    """Return a significance marker string for a Wilcoxon result."""
    if not wilcoxon_entry or wilcoxon_entry.get("p_value") is None:
        return ""
    p = wilcoxon_entry["p_value"]
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "(n.s.)"


def _ci_str(mean: float, ci: tuple | None) -> str:
    """Format mean ± CI as a compact string."""
    if not ci or ci == (0.0, 0.0):
        return f"{mean:.3f}"
    half = (ci[1] - ci[0]) / 2
    return f"{mean:.3f}±{half:.3f}"


def generate_markdown_report(
    scenario_name: str,
    summary: dict[str, dict],
    results: list[TrialResult],
    engine_method: str = "consensus_engine",
) -> str:
    lines = [
        f"# Evaluation Report: {scenario_name}",
        f"\nTotal trials: {len(results)} (raw data in `data/logs/`)",
        "",
        "> **Statistical note**: CIs are 95% bootstrap (1000 resamples). "
        "Significance markers (\\*, \\*\\*) are Wilcoxon signed-rank tests "
        f"comparing each method against `{engine_method}` (\\* p<0.05, \\*\\* p<0.01).",
        "",
        "## Summary by Method",
        "",
        "| Method | Agree% | Pareto Ratio (95% CI) | Nash Welfare (95% CI) | Min Util | Gini | Rounds | Privacy Cosine |",
        "|--------|--------|----------------------|----------------------|---------|------|--------|----------------|",
    ]

    for method, stats in sorted(summary.items()):
        # Significance marker vs engine
        wilcoxon_pareto = stats.get("wilcoxon_pareto_vs_engine")
        sig = _sig_marker(wilcoxon_pareto) if method != engine_method else "(engine)"

        pareto_str = _ci_str(stats["mean_pareto_ratio"], stats.get("ci95_pareto_ratio"))
        nash_str = _ci_str(stats["mean_nash_welfare"], stats.get("ci95_nash_welfare"))

        privacy_str = (
            f"{stats['mean_privacy_cosine']:.3f}"
            if stats.get("mean_privacy_cosine") is not None
            else "—"
        )

        lines.append(
            f"| {method} {sig} "
            f"| {stats['agreement_rate']:.1%} "
            f"| {pareto_str} "
            f"| {nash_str} "
            f"| {stats['mean_min_utility']:.3f} "
            f"| {stats['mean_gini']:.3f} "
            f"| {stats['mean_rounds']:.1f} "
            f"| {privacy_str} |"
        )

    lines.extend(["", "## Key Findings", ""])

    # Best methods
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

    # Privacy section
    privacy_methods = [(m, s) for m, s in summary.items() if s.get("mean_privacy_cosine") is not None]
    if privacy_methods:
        lines.extend(["", "## Privacy Leakage (Reconstruction Probe)", ""])
        lines.append(
            "Cosine similarity between probe-inferred weights and true weights. "
            "Uniform-random baseline ≈ 0.5–0.6 for 5 issues. Higher = more leakage."
        )
        lines.append("")
        for method, stats in privacy_methods:
            lines.append(f"- **{method}**: mean cosine = {stats['mean_privacy_cosine']:.4f}")

    # Bluff detection section
    bluff_methods = [(m, s) for m, s in summary.items() if s.get("bluff_detection_flagged", 0) > 0]
    if bluff_methods:
        lines.extend(["", "## Bluff Detection", ""])
        for method, stats in bluff_methods:
            n = stats["n_trials"]
            flagged = stats["bluff_detection_flagged"]
            lines.append(
                f"- **{method}**: bluff-suspected in {flagged}/{n} trials "
                f"({100*flagged/n:.0f}%)"
            )

    lines.extend([
        "",
        "---",
        "*Raw trial data: `data/logs/`. Re-aggregate with `src/eval/log_writer.load_trial_log()`.*",
    ])

    return "\n".join(lines)


def generate_json_report(
    scenario_name: str,
    summary: dict[str, dict],
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
