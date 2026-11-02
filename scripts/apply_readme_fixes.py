import re
from pathlib import Path

readme_path = Path("README.md")
with open(readme_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update Expected line
text = text.replace(
    "# Expected: 132 passed (+ 1 live integration test requiring GROQ_API_KEY)",
    "# Expected: 134 passed (+ 1 live integration test requiring GROQ_API_KEY)",
)

# 2. Update rate limiting & caps in API reference
old_limits = """> [!NOTE]
> **Public Endpoint Limits & Rate Limiting**:
> To protect compute resources, endpoints calling LLM inference enforce:
> - **Per-IP Rate Limit**: 10 requests per minute per client IP across `POST /api/negotiate`, `POST /api/negotiate/start`, `POST /api/scenario/parse`, and `POST /api/eval/run` (returns `429 Too Many Requests`).
> - **Negotiation Cap**: `max_rounds` capped at $\\le 10$ (returns `400 Bad Request` if exceeded).
> - **Evaluation Cap**: `n_trials` on public batch runs capped at $\\le 30$ (returns `400 Bad Request` if exceeded)."""

new_limits = """> [!NOTE]
> **Public Endpoint Limits & Rate Limiting**:
> To protect compute resources, endpoints calling LLM inference enforce:
> - **Per-IP Rate Limit**: 10 requests per minute per client IP across `POST /api/negotiate`, `POST /api/negotiate/start`, `POST /api/scenario/parse`, and `POST /api/eval/run` (returns `429 Too Many Requests`). Client IP is resolved from `X-Forwarded-For` (first hop) or `x-real-ip` when running behind reverse proxies (e.g. Vercel), falling back to `request.client.host`. Note that the in-memory limiter is per serverless instance and only best-effort.
> - **Negotiation Cap**: `max_rounds` capped at $\\le 10$ (returns `400 Bad Request` if exceeded).
> - **Evaluation Cap**: `n_trials` on public batch runs capped at $\\le 5$ (returns `400 Bad Request` if exceeded)."""

text = text.replace(old_limits.replace("\r\n", "\n"), new_limits.replace("\r\n", "\n"))
text = text.replace(old_limits, new_limits)

# 3. Update eval/run payload example
text = text.replace('"n_trials": 30,\n  "methods": ["naive_average"', '"n_trials": 5,\n  "methods": ["naive_average"')

# 4. Update Pareto / welfare ratio descriptions in evaluation overview
old_frontier_block = """### Utilitarian Welfare Ratio & Distance to Frontier

The **Utilitarian Welfare Ratio** measures the achieved sum of utilities relative to the maximum possible sum on the Pareto frontier:

$$\\text{UWR}(\\mathbf{x}^{\\ast}) = \\frac{\\sum_{i=1}^{N} U_i(\\mathbf{x}^{\\ast})}{\\displaystyle\\max_{\\mathbf{u} \\in \\mathcal{F}} \\sum_{i=1}^{N} u_i}$$

The **Distance to Frontier** measures the Euclidean distance from the achieved utility vector $\\mathbf{u}(\\mathbf{x}^{\\ast})$ to the nearest point on the Pareto frontier $\\mathcal{F}$ in $[0, 1]^N$ utility space:

$$\\text{DistToFrontier}(\\mathbf{x}^{\\ast}) = \\min_{\\mathbf{y} \\in \\mathcal{F}} \\|\\mathbf{u}(\\mathbf{x}^{\\ast}) - \\mathbf{y}\\|_2$$

where $\\mathcal{F}$ is the Pareto frontier computed by exhaustive grid search ($\\le 4$ issues) or Monte Carlo sampling (5–6 issues)."""

new_frontier_block = """### Headline Welfare Metrics: Nash Welfare & Utilitarian Welfare Ratio

Consensus Engine benchmarks headline **Nash Social Welfare** ($\\text{NSW}$) and **Utilitarian Welfare Ratio** ($\\text{UWR}$) as primary quality metrics.

The **Utilitarian Welfare Ratio** measures the achieved sum of utilities relative to the maximum possible sum on the Pareto frontier:

$$\\text{UWR}(\\mathbf{x}^{\\ast}) = \\frac{\\sum_{i=1}^{N} U_i(\\mathbf{x}^{\\ast})}{\\displaystyle\\max_{\\mathbf{u} \\in \\mathcal{F}} \\sum_{i=1}^{N} u_i}$$

The **Distance to Frontier** measures the Euclidean distance from the achieved utility vector $\\mathbf{u}(\\mathbf{x}^{\\ast})$ to the nearest point on the discrete Pareto frontier $\\mathcal{F}$ in $[0, 1]^N$ utility space:

$$\\text{DistToFrontier}(\\mathbf{x}^{\\ast}) = \\min_{\\mathbf{y} \\in \\mathcal{F}} \\|\\mathbf{u}(\\mathbf{x}^{\\ast}) - \\mathbf{y}\\|_2$$

where $\\mathcal{F}$ is the Pareto frontier computed by exhaustive grid search ($\\le 4$ issues) or Monte Carlo sampling (5–6 issues).

> ⚠️ **Metric Floor & Grid Resolution Caveat:**  
> Discrete grid evaluation in multi-issue spaces introduces an empirical discretization floor near ~0.035 for 5-issue domains. For example, `nash_bargaining` analytically maximises the Nash product and lies on the continuous Pareto frontier (continuous $\\text{DistToFrontier} \\approx 0$), but exhibits an empirical distance of ~0.0350 against the discrete grid. Therefore, **distance differences below the grid resolution (~0.035) are discretization artifacts and are not meaningful**. Nash social welfare and Utilitarian welfare ratio provide robust, continuous indicators of agreement quality."""

text = text.replace(old_frontier_block.replace("\r\n", "\n"), new_frontier_block.replace("\r\n", "\n"))
text = text.replace(old_frontier_block, new_frontier_block)

# 5. Update benchmark honest caveat
old_caveat = """> ⚠️ **What This Benchmark Measures (Honest Caveat):**  
> All 30 Consensus Engine trials ended in round 1 with exactly 4 LLM calls each (1 mediator proposal + 3 stakeholder critiques). Every party accepted the mediator's first proposal immediately. The engine's modest edge over the public midpoint (distance to frontier 0.0654 vs 0.0725, Wilcoxon $p = 0.0017^{**}$; Nash welfare $p = 0.0015^{**}$) therefore reflects the mediator adjusting the midpoint using issue descriptions, **NOT iterative negotiation**. This should not be described or interpreted as negotiation superiority. Results reflect one scenario (`business_deal`) and one model config (`20b_vs_20b`).
>
> In symmetric continuous domains without private reservation boundaries, arithmetic baselines like `naive_average` achieve high utilitarian welfare (0.9525) because the arithmetic mean is an effective centroid when preferences are symmetric. Note that `private_ideal_average` is an alias for `naive_average` (evaluated across 4 baselines, N=120 baseline trials; N=150 total across all methods)."""

new_caveat = """> ⚠️ **What This Benchmark Measures (Honest Caveat):**  
> All 30 Consensus Engine trials ended in round 1 with exactly 4 LLM calls each (1 mediator proposal + 3 stakeholder critiques). Every party accepted the mediator's first proposal immediately. The engine's edge over the public midpoint reflects the mediator adjusting the midpoint using issue descriptions, **NOT iterative negotiation**. This should not be described or interpreted as negotiation superiority.
>
> When comparing methods, **Nash social welfare and Utilitarian welfare ratio serve as headline metrics**. Because the discrete frontier grid sets an empirical distance floor near ~0.035 (under which even theoretical frontiers like `nash_bargaining` score 0.0350 despite continuous distance $\\approx 0$), distance differences below the grid resolution (~0.035) are discretization artifacts and not meaningful. In symmetric continuous domains without tight private reservations, arithmetic baselines like `naive_average` achieve high utilitarian welfare because the arithmetic mean is an effective centroid when preferences are symmetric."""

text = text.replace(old_caveat.replace("\r\n", "\n"), new_caveat.replace("\r\n", "\n"))
text = text.replace(old_caveat, new_caveat)

# 6. Remove is_approximate_frontier claim
old_approx_claim = "In reports and API responses, metrics derived from Monte Carlo frontiers are explicitly flagged with `is_approximate_frontier: true`."
new_approx_claim = "In reports and API responses, the strategy used is explicitly flagged via the `pareto_mode` field (`\"exhaustive\"` for ≤ 4 issues, `\"monte_carlo\"` for 5–6 issues)."
text = text.replace(old_approx_claim, new_approx_claim)

# 7. Replace "Pareto efficiency ratios" with "utilitarian welfare ratios"
text = text.replace("Pareto efficiency ratios", "utilitarian welfare ratios")

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Applied README text updates successfully")

