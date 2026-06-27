"""LangGraph shared state definition for the triage workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class TriageState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────
    run_id: str
    test_suite: str
    branch: str
    commit_sha: str
    raw_log: str
    git_diff: str

    # ── Agent outputs ──────────────────────────────────────────────────────
    log_analysis: dict[str, Any]       # from LogAnalyzerAgent
    diff_analysis: dict[str, Any]      # from DiffAnalyzerAgent
    rca_report: dict[str, Any]         # from RCASynthesizerAgent

    # ── Metadata ──────────────────────────────────────────────────────────
    errors: list[str]
    completed_nodes: list[str]
