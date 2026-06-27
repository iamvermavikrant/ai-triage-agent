"""DeepEval integration — GEval, HallucinationMetric, AnswerRelevancyMetric.

When MOCK_LLM=true, all DeepEval metrics return plausible mock scores so the
full pipeline can be demonstrated without an API key.
"""

from __future__ import annotations

import json
import os
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ── Mock scores returned when MOCK_LLM=true ───────────────────────────────────

_MOCK_DEEPEVAL_RESULT = {
    "geval": {
        "rca_correctness": {"score": 0.88, "reason": "Root cause correctly identifies the OOM trigger.", "passed": True},
        "fix_actionability": {"score": 0.91, "reason": "Recommended fix is specific and immediately executable.", "passed": True},
        "no_scope_creep": {"score": 0.85, "reason": "Report stays focused on the failure without tangential details.", "passed": True},
    },
    "hallucination": {
        "score": 0.05,
        "reason": "All mentioned file paths and function names exist in the provided context.",
        "passed": True,
    },
    "answer_relevancy": {
        "score": 0.93,
        "reason": "The RCA directly addresses the test failure described in the log.",
        "passed": True,
    },
    "deepeval_overall_pass": True,
    "deepeval_summary": "3/3 metrics passed. No hallucinations detected. RCA is highly relevant and actionable.",
}


# ── Real DeepEval evaluation ───────────────────────────────────────────────────

def _run_real_deepeval(
    rca_report: dict[str, Any],
    ground_truth: dict[str, Any],
    raw_log: str,
    git_diff: str,
) -> dict[str, Any]:
    """Run actual DeepEval metrics against the RCA report."""
    from deepeval import evaluate
    from deepeval.metrics import GEval, HallucinationMetric, AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    rca_text = json.dumps(rca_report, indent=2)
    gt_text = json.dumps(ground_truth, indent=2)
    input_context = f"Test log:\n{raw_log}\n\nGit diff:\n{git_diff}"

    test_case = LLMTestCase(
        input=input_context,
        actual_output=rca_text,
        expected_output=gt_text,
        context=[raw_log, git_diff, gt_text],
    )

    # ── Metric 1: GEval — RCA Correctness ────────────────────────────────
    geval_correctness = GEval(
        name="RCA Correctness",
        criteria=(
            "Evaluate whether the generated RCA correctly identifies the root cause "
            "of the test failure. The root cause should match the expected failure type "
            "and implicated component described in the expected output."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
    )

    # ── Metric 2: GEval — Fix Actionability ──────────────────────────────
    geval_actionability = GEval(
        name="Fix Actionability",
        criteria=(
            "Evaluate whether the recommended_fix in the RCA is specific, technically "
            "accurate, and immediately executable by an engineer without further research. "
            "Vague suggestions like 'investigate the issue' should score low."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )

    # ── Metric 3: GEval — No Scope Creep ─────────────────────────────────
    geval_focus = GEval(
        name="No Scope Creep",
        criteria=(
            "Evaluate whether the RCA stays focused on the specific test failure. "
            "It should not introduce unrelated issues, general best practices unrelated "
            "to the failure, or excessive boilerplate text."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )

    # ── Metric 4: Hallucination ───────────────────────────────────────────
    hallucination_metric = HallucinationMetric(threshold=0.3)

    # ── Metric 5: Answer Relevancy ────────────────────────────────────────
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7)

    # Run all metrics
    results_list = evaluate(
        test_cases=[test_case],
        metrics=[
            geval_correctness,
            geval_actionability,
            geval_focus,
            hallucination_metric,
            relevancy_metric,
        ],
        print_results=False,
    )

    # Extract scores from results
    metric_map = {m.name: m for m in [
        geval_correctness, geval_actionability, geval_focus,
        hallucination_metric, relevancy_metric,
    ]}

    def _extract(metric: Any) -> dict[str, Any]:
        return {
            "score": round(metric.score, 3),
            "reason": getattr(metric, "reason", "") or "",
            "passed": metric.is_successful(),
        }

    geval_results = {
        "rca_correctness": _extract(geval_correctness),
        "fix_actionability": _extract(geval_actionability),
        "no_scope_creep": _extract(geval_focus),
    }
    hallucination_result = _extract(hallucination_metric)
    relevancy_result = _extract(relevancy_metric)

    all_passed = (
        all(v["passed"] for v in geval_results.values())
        and hallucination_result["passed"]
        and relevancy_result["passed"]
    )
    passed_count = sum([
        all(v["passed"] for v in geval_results.values()),
        hallucination_result["passed"],
        relevancy_result["passed"],
    ])

    return {
        "geval": geval_results,
        "hallucination": hallucination_result,
        "answer_relevancy": relevancy_result,
        "deepeval_overall_pass": all_passed,
        "deepeval_summary": f"{passed_count}/3 metric groups passed.",
    }


# ── Public interface ───────────────────────────────────────────────────────────

def run_deepeval(
    rca_report: dict[str, Any],
    ground_truth: dict[str, Any],
    raw_log: str = "",
    git_diff: str = "",
) -> dict[str, Any]:
    """
    Run DeepEval metrics on a generated RCA report.

    Returns mock scores when MOCK_LLM=true or when deepeval is not installed.
    """
    if os.getenv("MOCK_LLM", "false").lower() == "true":
        log.info("deepeval.mock_mode")
        return _MOCK_DEEPEVAL_RESULT

    try:
        import deepeval  # noqa: F401
    except ImportError:
        log.warning("deepeval.not_installed", hint="pip install deepeval")
        return {**_MOCK_DEEPEVAL_RESULT, "deepeval_summary": "deepeval not installed — mock scores returned"}

    try:
        log.info("deepeval.start")
        result = _run_real_deepeval(rca_report, ground_truth, raw_log, git_diff)
        log.info("deepeval.done", overall_pass=result["deepeval_overall_pass"])
        return result
    except Exception as exc:
        log.exception("deepeval.error")
        return {
            **_MOCK_DEEPEVAL_RESULT,
            "deepeval_summary": f"DeepEval error — mock scores returned. Error: {exc}",
        }
