"""
Privacy Leakage Measurement
============================
After each negotiation, a "reconstruction probe" LLM is given the dialogue
transcript and asked to infer each agent's utility weight distribution.
The inferred weights are compared to ground-truth weights using:
  - Cosine similarity (1.0 = perfect inference, 0.0 = orthogonal)
  - Lift over uniform baseline (cosine - uniform_cosine)
  - Random Dirichlet reference baseline
  - KL divergence (0.0 = identical distribution, higher = more private)

Reference Baselines:
  - Uniform-guess baseline: cosine similarity between true weights and uniform weights [1/K, ..., 1/K].
    (On 5 issues with Dirichlet weights, this is ~0.80, NOT 1/sqrt(K) = 0.447).
  - Random-Dirichlet baseline: expected cosine similarity between true weights and random Dirichlet draws.
  - Leakage is reported as lift over the uniform baseline.

Robustness:
  - When the probe fails, it does NOT silently fall back to uniform weights;
    it returns None / probe_failed=True, and is excluded from means.
  - Runs on both (a) public transcript only and (b) transcript including private reasoning metadata.
"""
from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING, Optional, Any

import numpy as np

from src.llm.client import plain_completion
from src.config import settings
from src.models.evaluation import PrivacyMetrics, PrivacyVariantMetrics

if TYPE_CHECKING:
    from src.models.negotiation import NegotiationMessage
    from src.models.utility import StakeholderProfile

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two weight vectors."""
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


def _kl_divergence(p: list[float], q: list[float], epsilon: float = 1e-9) -> float:
    """
    KL divergence D(P || Q) where P = true distribution, Q = inferred.
    Higher values mean the inferred distribution is further from truth.
    """
    p_arr = np.array(p, dtype=float)
    q_arr = np.array(q, dtype=float)
    p_arr = p_arr / (p_arr.sum() + epsilon)
    q_arr = q_arr / (q_arr.sum() + epsilon)
    p_arr = np.clip(p_arr, epsilon, 1.0)
    q_arr = np.clip(q_arr, epsilon, 1.0)
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


def compute_reference_baselines(
    true_weights: dict[str, dict[str, float]],
    issues: list[dict],
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute two reference baselines across all agents:
      1. uniform-guess cosine: cosine(true, [1/K, ..., 1/K])
      2. random-Dirichlet-guess cosine: E[cosine(true, dirichlet_random)]
    """
    issue_names = [i["name"] for i in issues]
    k = len(issue_names)
    if k == 0 or not true_weights:
        return (0.0, 0.0)

    uniform_vec = [1.0 / k] * k
    uniform_cosines = []
    for weights in true_weights.values():
        true_vec = [weights.get(iss, 0.0) for iss in issue_names]
        uniform_cosines.append(_cosine_similarity(true_vec, uniform_vec))

    mean_uniform = float(np.mean(uniform_cosines)) if uniform_cosines else 0.0

    rng = np.random.RandomState(seed)
    dirichlet_cosines = []
    for weights in true_weights.values():
        true_vec = [weights.get(iss, 0.0) for iss in issue_names]
        samples = rng.dirichlet(np.ones(k), size=n_samples)
        for s in samples:
            dirichlet_cosines.append(_cosine_similarity(true_vec, s.tolist()))

    mean_dirichlet = float(np.mean(dirichlet_cosines)) if dirichlet_cosines else 0.0
    return (round(mean_uniform, 4), round(mean_dirichlet, 4))


def _build_probe_prompt(
    transcript_text: str,
    agent_names: list[str],
    issue_names: list[str],
) -> str:
    return f"""You are a privacy analyst. Below is a complete negotiation transcript.
Your task: infer each agent's utility weight distribution over the negotiation issues,
based ONLY on what the agents said in the transcript (their reasoning, demands, and concessions).

Do NOT use any knowledge outside this transcript.

Issues to weight over: {json.dumps(issue_names)}
Agents to analyze: {json.dumps(agent_names)}

TRANSCRIPT:
{transcript_text}

Respond with a JSON object in this exact structure:
{{
  "<agent_name>": {{
    "<issue_name>": <weight_0_to_1>,
    ...
  }},
  ...
}}

Weights for each agent should sum to approximately 1.0.
Make your best estimate based on the linguistic evidence in the transcript.
If you cannot infer a weight, distribute remaining weight equally."""


def _format_transcript(messages: list, include_reasoning: bool = False) -> str:
    """Convert NegotiationMessage list to a readable transcript string."""
    lines = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            m = msg.model_dump()
        else:
            m = msg
        role = m.get("role", "?")
        agent = m.get("agent_name", "?")
        rnd = m.get("round_number", 0)
        content = m.get("content", "")
        line = f"[Round {rnd}] {agent} ({role}): {content}"
        if include_reasoning:
            meta = m.get("metadata", {})
            reasoning = meta.get("reasoning", "") if isinstance(meta, dict) else ""
            if reasoning:
                line += f"\n  Reasoning: {reasoning}"
        lines.append(line)
    return "\n".join(lines)


def reconstruct_weights(
    transcript: list,
    agent_names: list[str],
    issues: list[dict],
    model: str | None = None,
    include_reasoning: bool = False,
) -> Optional[dict[str, dict[str, float]]]:
    """
    Ask a probe LLM to reconstruct utility weights from the transcript.

    Returns:
        Dict mapping agent_name → {issue_name → inferred_weight},
        or None if the probe call/parse fails (no silent fallback).
    """
    model = model or settings.negotiator_model
    issue_names = [i["name"] for i in issues]
    transcript_text = _format_transcript(transcript, include_reasoning=include_reasoning)

    if not transcript_text.strip():
        logger.warning("privacy probe: empty transcript")
        return None

    prompt = _build_probe_prompt(transcript_text, agent_names, issue_names)

    try:
        raw = plain_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a privacy analysis tool. You MUST respond with valid JSON only — "
                        "no explanation, no markdown, just the JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.1,
        )
        inferred = json.loads(raw)
        result = {}
        for name in agent_names:
            if name not in inferred:
                continue
            agent_weights = inferred[name]
            total = sum(agent_weights.values()) or 1.0
            result[name] = {
                iss: round(agent_weights.get(iss, 1.0 / len(issue_names)) / total, 4)
                for iss in issue_names
            }
        return result if result else None
    except Exception as e:
        logger.warning("privacy probe reconstruction failed: %s", e)
        return None


def compute_leakage_scores(
    true_weights: dict[str, dict[str, float]],
    inferred_weights: Optional[dict[str, dict[str, float]]],
    issues: list[dict],
) -> dict[str, dict[str, float]]:
    """Compute per-agent leakage metrics."""
    if inferred_weights is None:
        return {}
    issue_names = [i["name"] for i in issues]
    scores = {}
    for name in true_weights:
        if name not in inferred_weights:
            continue
        true_vec = [true_weights[name].get(iss, 0.0) for iss in issue_names]
        inf_vec = [inferred_weights[name].get(iss, 0.0) for iss in issue_names]
        cos_sim = _cosine_similarity(true_vec, inf_vec)
        kl_div = _kl_divergence(true_vec, inf_vec)
        scores[name] = {
            "cosine_similarity": round(cos_sim, 4),
            "kl_divergence": round(kl_div, 4),
            "leakage_score": round(cos_sim, 4),
        }
    return scores


def measure_privacy_variant(
    transcript: list,
    profiles: list,
    issues: list[dict],
    uniform_baseline_cosine: float,
    model: str | None = None,
    include_reasoning: bool = False,
) -> PrivacyVariantMetrics:
    agent_names = [p.name for p in profiles]
    true_weights = {p.name: p.utility_function.weights for p in profiles}

    inferred = reconstruct_weights(
        transcript=transcript,
        agent_names=agent_names,
        issues=issues,
        model=model,
        include_reasoning=include_reasoning,
    )

    if inferred is None:
        return PrivacyVariantMetrics(
            mean_cosine_similarity=None,
            lift_over_uniform=None,
            mean_kl_divergence=None,
            per_agent_leakage={},
            failure_count=len(agent_names),
            probe_failed=True,
        )

    per_agent_scores = compute_leakage_scores(true_weights, inferred, issues)
    cosines = [s["cosine_similarity"] for s in per_agent_scores.values()]
    kls = [s["kl_divergence"] for s in per_agent_scores.values()]

    failure_count = len(agent_names) - len(per_agent_scores)
    probe_failed = len(per_agent_scores) == 0

    if cosines:
        mean_cos = round(float(np.mean(cosines)), 4)
        mean_kl = round(float(np.mean(kls)), 4)
        lift = round(mean_cos - uniform_baseline_cosine, 4)
    else:
        mean_cos = None
        mean_kl = None
        lift = None

    return PrivacyVariantMetrics(
        mean_cosine_similarity=mean_cos,
        lift_over_uniform=lift,
        mean_kl_divergence=mean_kl,
        per_agent_leakage=per_agent_scores,
        failure_count=failure_count,
        probe_failed=probe_failed,
    )


def measure_privacy_leakage(
    transcript: list,
    profiles: list,
    issues: list[dict],
    model: str | None = None,
) -> PrivacyMetrics:
    """
    Measures privacy leakage across:
      (a) public transcript only
      (b) transcript with reasoning metadata
    Also computes reference baselines (uniform guess and Dirichlet random).
    """
    agent_names = [p.name for p in profiles]
    true_weights = {p.name: p.utility_function.weights for p in profiles}

    uniform_base, dirichlet_base = compute_reference_baselines(true_weights, issues)

    public_metrics = measure_privacy_variant(
        transcript=transcript,
        profiles=profiles,
        issues=issues,
        uniform_baseline_cosine=uniform_base,
        model=model,
        include_reasoning=False,
    )

    reasoning_metrics = measure_privacy_variant(
        transcript=transcript,
        profiles=profiles,
        issues=issues,
        uniform_baseline_cosine=uniform_base,
        model=model,
        include_reasoning=True,
    )

    return PrivacyMetrics(
        mean_cosine_similarity=public_metrics.mean_cosine_similarity,
        mean_kl_divergence=public_metrics.mean_kl_divergence,
        per_agent_leakage=public_metrics.per_agent_leakage,
        n_agents=len(profiles),
        n_issues=len(issues),
        uniform_baseline_cosine=uniform_base,
        dirichlet_baseline_cosine=dirichlet_base,
        lift_over_uniform=public_metrics.lift_over_uniform,
        failure_count=public_metrics.failure_count,
        probe_failed=public_metrics.probe_failed,
        public_transcript=public_metrics,
        with_reasoning=reasoning_metrics,
    )


def run_privacy_probe(
    outcome: Any,
    profiles: list,
    issues: list[dict],
    model: str | None = None,
) -> Optional[PrivacyMetrics]:
    """Helper alias for benchmark runner."""
    messages = getattr(outcome, "messages", None)
    if not messages:
        return None
    return measure_privacy_leakage(
        transcript=messages,
        profiles=profiles,
        issues=issues,
        model=model,
    )
