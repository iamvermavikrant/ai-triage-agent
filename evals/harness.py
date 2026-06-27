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


def run_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    """Execute one fixture end-to-end and return the scored result."""
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

    # ── Judge ───────────────────────────────────────────────────────────
    if rca_report and not errors:
        judgment = judge_rca(rca_report, fixture["ground_truth"])
    else:
        judgment = {
            "scores": {},
            "weighted_total": 0.0,
            "pass": False,
            "critique": f"Pipeline errors: {errors}",
        }

    return {
        "fixture_id": fid,
        "description": fixture.get("description", ""),
        "elapsed_s": elapsed,
        "rca_report": rca_report,
        "judgment": judgment,
        "pipeline_errors": errors,
    }


def run_all_evals(fixtures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run the full eval suite and return an aggregate report."""
    all_fixtures = fixtures or load_fixtures()
    results = []

    for fixture in all_fixtures:
        try:
            result = run_fixture(fixture)
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
                "pipeline_errors": [str(exc)],
            }
        results.append(result)

    total = len(results)
    passed = sum(1 for r in results if r["judgment"]["pass"])
    avg_score = (
        sum(r["judgment"]["weighted_total"] for r in results) / total if total else 0.0
    )

    return {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 3) if total else 0.0,
            "avg_score": round(avg_score, 2),
        },
        "results": results,
    }


if __name__ == "__main__":
    import structlog
    structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

    report = run_all_evals()
    print_report(report)
    save_report(report)
