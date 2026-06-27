"""Log Analyzer Agent — extracts structured failure signals from raw CI logs."""

from __future__ import annotations

import structlog

from ai_triage_agent.graph.state import TriageState
from ai_triage_agent.utils.llm_client import call_llm_json
from ai_triage_agent.utils.prompt_loader import get_system_prompt, render_user_prompt

log = structlog.get_logger(__name__)


def log_analyzer_node(state: TriageState) -> TriageState:
    """LangGraph node: analyze raw_log → log_analysis."""
    log.info("log_analyzer.start", run_id=state.get("run_id"))

    raw_log = state.get("raw_log", "")
    if not raw_log:
        return {**state, "errors": [*(state.get("errors") or []), "log_analyzer: empty raw_log"]}

    system = get_system_prompt("log_analyzer")
    user = render_user_prompt("log_analyzer", log_content=raw_log)

    analysis = call_llm_json(system=system, user=user)

    log.info(
        "log_analyzer.done",
        failure_type=analysis.get("failure_type"),
        severity=analysis.get("severity"),
    )
    completed = [*(state.get("completed_nodes") or []), "log_analyzer"]
    return {**state, "log_analysis": analysis, "completed_nodes": completed}
