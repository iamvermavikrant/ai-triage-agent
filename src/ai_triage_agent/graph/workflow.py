"""LangGraph workflow — wires Log Analyzer → Diff Analyzer → RCA Synthesizer."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from ai_triage_agent.agents.diff_analyzer import diff_analyzer_node
from ai_triage_agent.agents.log_analyzer import log_analyzer_node
from ai_triage_agent.agents.rca_synthesizer import rca_synthesizer_node
from ai_triage_agent.graph.state import TriageState


def _has_errors(state: TriageState) -> str:
    """Conditional edge: if errors exist, skip to END."""
    return "end" if state.get("errors") else "continue"


def build_triage_graph() -> StateGraph:
    """Construct and compile the triage LangGraph."""
    builder = StateGraph(TriageState)

    builder.add_node("log_analyzer", log_analyzer_node)
    builder.add_node("diff_analyzer", diff_analyzer_node)
    builder.add_node("rca_synthesizer", rca_synthesizer_node)

    builder.set_entry_point("log_analyzer")

    builder.add_conditional_edges(
        "log_analyzer",
        _has_errors,
        {"end": END, "continue": "diff_analyzer"},
    )
    builder.add_conditional_edges(
        "diff_analyzer",
        _has_errors,
        {"end": END, "continue": "rca_synthesizer"},
    )
    builder.add_edge("rca_synthesizer", END)

    return builder.compile()


# Module-level singleton — compiled once, reused across invocations.
triage_graph = build_triage_graph()
