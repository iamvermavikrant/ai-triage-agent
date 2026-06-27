"""LLM-as-judge: scores RCA quality against ground truth."""

from __future__ import annotations

import json
import os
from typing import Any

import structlog

from ai_triage_agent.utils.llm_client import call_llm_json
from ai_triage_agent.utils.prompt_loader import get_system_prompt, render_user_prompt

log = structlog.get_logger(__name__)

PASS_THRESHOLD = 7.0

DIMENSION_WEIGHTS = {
    "accuracy": 0.35,
    "completeness": 0.20,
    "actionability": 0.25,
    "precision": 0.10,
    "clarity": 0.10,
}


def judge_rca(
    generated_rca: dict[str, Any],
    ground_truth: dict[str, Any],
    judge_model: str | None = None,
) -> dict[str, Any]:
    """
    Score a generated RCA against the ground truth using an LLM judge.

    Returns:
        {
            "scores": {...},
            "weighted_total": float,
            "pass": bool,
            "critique": str,
            "raw_judge_response": {...}
        }
    """
    model = judge_model or os.getenv("JUDGE_MODEL", "claude-opus-4-8")
    system = get_system_prompt("llm_judge")
    user = render_user_prompt(
        "llm_judge",
        ground_truth=json.dumps(ground_truth, indent=2),
        generated_rca=json.dumps(generated_rca, indent=2),
    )

    log.info("judge.start", model=model)
    raw = call_llm_json(system=system, user=user, model=model)

    scores = raw.get("scores", {})
    weighted_total = sum(
        scores.get(dim, 0) * weight for dim, weight in DIMENSION_WEIGHTS.items()
    )
    passed = weighted_total >= PASS_THRESHOLD

    log.info("judge.done", weighted_total=round(weighted_total, 2), passed=passed)

    return {
        "scores": scores,
        "weighted_total": round(weighted_total, 2),
        "pass": passed,
        "critique": raw.get("critique", ""),
        "raw_judge_response": raw,
    }
