"""Eval harness — runs all fixtures through the triage pipeline and judges results."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog

from ai_triage_agent.graph.workflow import triage_graph
from ai_triage_agent.mcp.tools.fetch_test_logs import fetch_test_logs
from ai_triage_agent.mcp.tools.get_git_diff import get_git_diff
from evals.deepeval_metrics import run_deepeval
from evals.judge import judge_rca
from evals.report import print_report, save_report

log = structlog.get_logger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixtures() -> list[dict[str, Any]]:
    fixtures = []
    for path in sorted(FIXTURES_DIR.glob("fixture_*.json")):
        with path.open() as fh:
            fixtures.append(json.load(fh))
    return fixtures


def run_fixture(
    fixture: dict[str, Any],
    run_custom: bool = True,
    run_deepeval_judge: bool = True,
) -> dict[str, Any]:
    """Execute one fixture end-to-end and return scores from selected judges.

    Args:
        fixture: Loaded fixture dict.
        run_custom: Whether to run the custom LLM-as-judge.
        run_deepeval_judge: Whether to run DeepEval metrics.
    """
    inp = fixture["input"]
    fid = fixture["id"]
    log.info("harness.fixture_start", id=fid)

    # ── Fetch inputs via MCP tools ──────────────────────────────────────
    raw_log = fetch_test_logs(
        run_id=inp["run_id"],
        backend=inp.get("backend_log", "mock"),
    )
    git_diff = get_git_diff(
        commit_sha=inp["commit_sha"],
        backend=inp.get("backend_diff", "mock"),
    )

    # ── Run triage graph ────────────────────────────────────────────────
    t0 = time.perf_counter()
    final_state = triage_graph.invoke(
        {
            "run_id": fid,
            "test_suite": inp["test_suite"],
            "branch": inp["branch"],
            "commit_sha": inp["commit_sha"],
            "raw_log": raw_log,
            "git_diff": git_diff,
            "errors": [],
            "completed_nodes": [],
        }
    )
    elapsed = round(time.perf_counter() - t0, 2)

    rca_report = final_state.get("rca_report", {})
    errors = final_state.get("errors", [])

    _empty_judgment = {
        "scores": {},
        "weighted_total": 0.0,
        "pass": False,
        "critique": f"Pipeline errors: {errors}",
    }
    _empty_deepeval = {
        "geval": {},
        "hallucination": {},
        "answer_relevancy": {},
        "deepeval_overall_pass": False,
        "deepeval_summary": f"Skipped — pipeline errors: {errors}",
    }

    judgment = _empty_judgment
    deepeval_result = _empty_deepeval

    if rca_report and not errors:
        if run_custom:
            # ── Judge 1: custom LLM-as-judge ───────────────────────────
            judgment = judge_rca(rca_report, fixture["ground_truth"])

        if run_deepeval_judge:
            # ── Judge 2: DeepEval metrics ───────────────────────────────
            deepeval_result = run_deepeval(
                rca_report=rca_report,
                ground_truth=fixture["ground_truth"],
                raw_log=raw_log,
                git_diff=git_diff,
            )

    return {
        "fixture_id": fid,
        "description": fixture.get("description", ""),
        "elapsed_s": elapsed,
        "rca_report": rca_report,
        "judgment": judgment,
        "deepeval": deepeval_result,
        "pipeline_errors": errors,
    }


def run_all_evals(
    fixtures: list[dict[str, Any]] | None = None,
    run_custom: bool = True,
    run_deepeval_judge: bool = True,
) -> dict[str, Any]:
    """Run the full eval suite and return an aggregate report.

    Args:
        fixtures: Override fixture list (defaults to all fixtures/ files).
        run_custom: Enable the custom LLM-as-judge scorer.
        run_deepeval_judge: Enable DeepEval metrics.
    """
    all_fixtures = fixtures or load_fixtures()
    results = []

    for fixture in all_fixtures:
        try:
            result = run_fixture(
                fixture,
                run_custom=run_custom,
                run_deepeval_judge=run_deepeval_judge,
            )
        except Exception as exc:
            log.exception("harness.fixture_error", id=fixture.get("id"))
            result = {
                "fixture_id": fixture.get("id"),
                "description": fixture.get("description", ""),
                "elapsed_s": 0.0,
                "rca_report": {},
                "judgment": {
                    "scores": {},
                    "weighted_total": 0.0,
                    "pass": False,
                    "critique": f"Unhandled exception: {exc}",
                },
                "deepeval": {
                    "geval": {},
                    "hallucination": {},
                    "answer_relevancy": {},
                    "deepeval_overall_pass": False,
                    "deepeval_summary": f"Unhandled exception: {exc}",
                },
                "pipeline_errors": [str(exc)],
            }
        results.append(result)

    total = len(results)

    # ── Custom judge aggregate ──────────────────────────────────────────
    custom_passed = sum(1 for r in results if r["judgment"]["pass"])
    avg_score = (
        sum(r["judgment"]["weighted_total"] for r in results) / total if total else 0.0
    )

    # ── DeepEval aggregate ──────────────────────────────────────────────
    deepeval_passed = sum(1 for r in results if r["deepeval"]["deepeval_overall_pass"])

    return {
        "summary": {
            "total": total,
            # Custom judge
            "custom_passed": custom_passed,
            "custom_failed": total - custom_passed,
            "custom_pass_rate": round(custom_passed / total, 3) if total else 0.0,
            "custom_avg_score": round(avg_score, 2),
            # DeepEval
            "deepeval_passed": deepeval_passed,
            "deepeval_failed": total - deepeval_passed,
            "deepeval_pass_rate": round(deepeval_passed / total, 3) if total else 0.0,
        },
        "results": results,
    }


if __name__ == "__main__":
    import argparse
    import structlog

    structlog.configure(
        processors=[structlog.stdlib.add_log_level, structlog.processors.KeyValueRenderer()]
    )

    parser = argparse.ArgumentParser(description="AI Triage Agent — Eval Harness")
    parser.add_argument(
        "--judge",
        choices=["custom", "deepeval", "both"],
        default="both",
        help=(
            "Which judge(s) to run:\n"
            "  custom   — custom LLM-as-judge only (5 weighted dimensions)\n"
            "  deepeval — DeepEval metrics only (GEval + Hallucination + Relevancy)\n"
            "  both     — run both judges (default)"
        ),
    )
    args = parser.parse_args()

    run_custom = args.judge in ("custom", "both")
    run_deepeval_judge = args.judge in ("deepeval", "both")

    report = run_all_evals(run_custom=run_custom, run_deepeval_judge=run_deepeval_judge)
    print_report(report)
    save_report(report)
