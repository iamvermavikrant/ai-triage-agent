"""RCA Synthesizer Agent — produces the final root-cause analysis report."""

from __future__ import annotations

import json

import structlog

from ai_triage_agent.graph.state import TriageState
from ai_triage_agent.utils.llm_client import call_llm_json
from ai_triage_agent.utils.prompt_loader import get_system_prompt, render_user_prompt

log = structlog.get_logger(__name__)


def rca_synthesizer_node(state: TriageState) -> TriageState:
    """LangGraph node: (log_analysis, diff_analysis) → rca_report."""
    log.info("rca_synthesizer.start", run_id=state.get("run_id"))

    log_analysis = state.get("log_analysis")
    diff_analysis = state.get("diff_analysis")

    missing = []
    if not log_analysis:
        missing.append("log_analysis")
    if not diff_analysis:
        missing.append("diff_analysis")

    if missing:
        return {
            **state,
            "errors": [*(state.get("errors") or []), f"rca_synthesizer: missing {missing}"],
        }

    system = get_system_prompt("rca_synthesizer")
    user = render_user_prompt(
        "rca_synthesizer",
        log_analysis=json.dumps(log_analysis, indent=2),
        diff_analysis=json.dumps(diff_analysis, indent=2),
        test_suite=state.get("test_suite", "unknown"),
        commit_sha=state.get("commit_sha", "unknown"),
        branch=state.get("branch", "unknown"),
    )

    report = call_llm_json(system=system, user=user, max_tokens=4096)

    log.info(
        "rca_synthesizer.done",
        priority=report.get("priority"),
        title=report.get("title"),
    )
    completed = [*(state.get("completed_nodes") or []), "rca_synthesizer"]
    return {**state, "rca_report": report, "completed_nodes": completed}
