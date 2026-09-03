"""
Privacy Leakage Measurement
============================
After each negotiation, a "reconstruction probe" LLM is given the full dialogue
transcript and asked to infer each agent's utility weight distribution.
The inferred weights are compared to the ground-truth weights using:
  - Cosine similarity  (1.0 = perfect inference, 0.0 = orthogonal)
  - KL divergence      (0.0 = identical distribution, higher = more private)

A cosine similarity near 1.0 means the transcript leaked almost full weight
information. A cosine similarity near 0.5–0.6 (random baseline for 5 issues)
indicates privacy was preserved.

Usage:
    from src.eval.privacy import measure_privacy_leakage

    leakage = measure_privacy_leakage(
        transcript=outcome.messages,
        profiles=profiles,
        issues=issues,
    )
    print(leakage)  # PrivacyMetrics
"""
from __future__ import annotations
import json
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from src.llm.client import plain_completion
from src.config import settings

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
    Higher values mean the inferred distribution is further from truth
    (i.e., more privacy preserved).
    """
    p_arr = np.array(p, dtype=float)
    q_arr = np.array(q, dtype=float)
    # Normalize to valid probability distributions
    p_arr = p_arr / (p_arr.sum() + epsilon)
    q_arr = q_arr / (q_arr.sum() + epsilon)
    p_arr = np.clip(p_arr, epsilon, 1.0)
    q_arr = np.clip(q_arr, epsilon, 1.0)
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


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


def _format_transcript(messages: list) -> str:
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
        # Include reasoning from metadata if available
        meta = m.get("metadata", {})
        reasoning = meta.get("reasoning", "") if isinstance(meta, dict) else ""
        line = f"[Round {rnd}] {agent} ({role}): {content}"
        if reasoning:
            line += f"\n  Reasoning: {reasoning}"
        lines.append(line)
    return "\n".join(lines)


def reconstruct_weights(
    transcript: list,
    agent_names: list[str],
    issues: list[dict],
    model: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Ask a probe LLM to reconstruct utility weights from the transcript.

    Returns:
        Dict mapping agent_name → {issue_name → inferred_weight}
        Returns empty dicts per agent on failure.
    """
    model = model or settings.negotiator_model
    issue_names = [i["name"] for i in issues]
    transcript_text = _format_transcript(transcript)

    if not transcript_text.strip():
        logger.warning("privacy probe: empty transcript, returning uniform weights")
        uniform = 1.0 / len(issue_names) if issue_names else 0.0
        return {name: {iss: uniform for iss in issue_names} for name in agent_names}

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
            temperature=0.1,  # Low temperature for more deterministic inference
        )
        inferred = json.loads(raw)
        # Normalise and validate
        result = {}
        for name in agent_names:
            agent_weights = inferred.get(name, {})
            total = sum(agent_weights.values()) or 1.0
            result[name] = {
                iss: round(agent_weights.get(iss, 1.0 / len(issue_names)) / total, 4)
                for iss in issue_names
            }
        return result
    except Exception as e:
        logger.warning("privacy probe reconstruction failed: %s", e)
        uniform = 1.0 / len(issue_names) if issue_names else 0.0
        return {name: {iss: uniform for iss in issue_names} for name in agent_names}


def compute_leakage_scores(
    true_weights: dict[str, dict[str, float]],
    inferred_weights: dict[str, dict[str, float]],
    issues: list[dict],
) -> dict[str, dict[str, float]]:
    """
    Compute per-agent leakage metrics.

    Returns:
        Dict mapping agent_name → {cosine_similarity, kl_divergence, leakage_score}
        leakage_score = cosine_similarity (higher = more leakage)
    """
    issue_names = [i["name"] for i in issues]
    scores = {}
    for name in true_weights:
        true_vec = [true_weights[name].get(iss, 0.0) for iss in issue_names]
        inf_vec = [inferred_weights.get(name, {}).get(iss, 0.0) for iss in issue_names]
        cos_sim = _cosine_similarity(true_vec, inf_vec)
        kl_div = _kl_divergence(true_vec, inf_vec)
        scores[name] = {
            "cosine_similarity": round(cos_sim, 4),
            "kl_divergence": round(kl_div, 4),
            "leakage_score": round(cos_sim, 4),  # cosine is the headline metric
        }
        logger.info(
            "privacy_leakage: agent=%s cosine=%.4f kl=%.4f",
            name, cos_sim, kl_div,
        )
    return scores


def measure_privacy_leakage(
    transcript: list,
    profiles: list,
    issues: list[dict],
    model: str | None = None,
) -> "PrivacyMetrics":
    """
    Full pipeline: reconstruct weights from transcript, compare to ground truth,
    return PrivacyMetrics.

    Args:
        transcript: List of NegotiationMessage objects or dicts
        profiles: List of StakeholderProfile objects (provides ground truth weights)
        issues: Issue definitions
        model: Override LLM model for the probe (defaults to negotiator_model)
    """
    from src.models.evaluation import PrivacyMetrics  # local import to avoid circular

    agent_names = [p.name for p in profiles]
    true_weights = {
        p.name: p.utility_function.weights for p in profiles
    }

    inferred_weights = reconstruct_weights(transcript, agent_names, issues, model=model)
    per_agent_scores = compute_leakage_scores(true_weights, inferred_weights, issues)

    all_cosine = [s["cosine_similarity"] for s in per_agent_scores.values()]
    all_kl = [s["kl_divergence"] for s in per_agent_scores.values()]

    return PrivacyMetrics(
        mean_cosine_similarity=round(float(np.mean(all_cosine)), 4) if all_cosine else 0.0,
        mean_kl_divergence=round(float(np.mean(all_kl)), 4) if all_kl else 0.0,
        per_agent_leakage=per_agent_scores,
        n_agents=len(profiles),
        n_issues=len(issues),
    )
