"""Diff Analyzer Agent — correlates git diff with failure signals."""

from __future__ import annotations

import json

import structlog

from ai_triage_agent.graph.state import TriageState
from ai_triage_agent.utils.llm_client import call_llm_json
from ai_triage_agent.utils.prompt_loader import get_system_prompt, render_user_prompt

log = structlog.get_logger(__name__)


def diff_analyzer_node(state: TriageState) -> TriageState:
    """LangGraph node: (log_analysis, git_diff) → diff_analysis."""
    log.info("diff_analyzer.start", run_id=state.get("run_id"))

    log_analysis = state.get("log_analysis")
    git_diff = state.get("git_diff", "")

    if not log_analysis:
        return {**state, "errors": [*(state.get("errors") or []), "diff_analyzer: missing log_analysis"]}

    if not git_diff:
        # No diff available — return low-risk placeholder
        placeholder = {
            "implicated_files": [],
            "change_risk": "LOW",
            "regression_likely": False,
            "confidence": 0.0,
        }
        log.warning("diff_analyzer.no_diff")
        completed = [*(state.get("completed_nodes") or []), "diff_analyzer"]
        return {**state, "diff_analysis": placeholder, "completed_nodes": completed}

    system = get_system_prompt("diff_analyzer")
    user = render_user_prompt(
        "diff_analyzer",
        failure_signal=json.dumps(log_analysis, indent=2),
        git_diff=git_diff,
    )

    analysis = call_llm_json(system=system, user=user)

    log.info(
        "diff_analyzer.done",
        change_risk=analysis.get("change_risk"),
        regression_likely=analysis.get("regression_likely"),
        confidence=analysis.get("confidence"),
    )
    completed = [*(state.get("completed_nodes") or []), "diff_analyzer"]
    return {**state, "diff_analysis": analysis, "completed_nodes": completed}
