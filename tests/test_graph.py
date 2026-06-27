"""Integration tests for the LangGraph triage workflow."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_triage_agent.graph.workflow import build_triage_graph


_MOCK_LOG_ANALYSIS = {
    "failure_type": "IMPORT_ERROR",
    "error_summary": "Cannot import FlashAttentionKernel.",
    "stack_frames": ["attention.py:5", "test_inference.py:3"],
    "affected_modules": ["src/models/attention.py"],
    "severity": "CRITICAL",
    "reproducible": True,
    "keywords": ["ImportError", "FlashAttentionKernel"],
}

_MOCK_DIFF_ANALYSIS = {
    "implicated_files": [
        {
            "file": "src/models/attention.py",
            "hunk_summary": "Changed import symbol",
            "relevance_score": 1.0,
            "reasoning": "Direct import rename caused the error.",
        }
    ],
    "change_risk": "HIGH",
    "regression_likely": True,
    "confidence": 0.99,
}

_MOCK_RCA = {
    "title": "ImportError: FlashAttentionKernel not found",
    "root_cause": "attention.py import was renamed to a non-existent symbol.",
    "contributing_factors": ["No import smoke test in CI"],
    "blast_radius": "All inference tests",
    "recommended_fix": "Revert import in src/models/attention.py.",
    "preventive_measures": ["Add import validation test"],
    "priority": "P0",
    "estimated_fix_time": "30m",
    "owner_hint": "ml-inference team",
}


class TestTriageGraph:
    def test_full_pipeline_happy_path(self):
        graph = build_triage_graph()
        with (
            patch("ai_triage_agent.agents.log_analyzer.call_llm_json", return_value=_MOCK_LOG_ANALYSIS),
            patch("ai_triage_agent.agents.diff_analyzer.call_llm_json", return_value=_MOCK_DIFF_ANALYSIS),
            patch("ai_triage_agent.agents.rca_synthesizer.call_llm_json", return_value=_MOCK_RCA),
        ):
            result = graph.invoke({
                "run_id": "test_run_001",
                "test_suite": "test_inference",
                "branch": "refactor/attention-v2",
                "commit_sha": "def56789",
                "raw_log": "ERROR: ImportError",
                "git_diff": "diff --git a/attention.py...",
                "errors": [],
                "completed_nodes": [],
            })

        assert result["rca_report"]["priority"] == "P0"
        assert result["log_analysis"]["failure_type"] == "IMPORT_ERROR"
        assert "log_analyzer" in result["completed_nodes"]
        assert "diff_analyzer" in result["completed_nodes"]
        assert "rca_synthesizer" in result["completed_nodes"]
        assert not result.get("errors")

    def test_pipeline_short_circuits_on_empty_log(self):
        graph = build_triage_graph()
        result = graph.invoke({
            "run_id": "test_run_002",
            "raw_log": "",
            "errors": [],
            "completed_nodes": [],
        })
        assert result.get("errors")
        assert result.get("rca_report") is None

    def test_completed_nodes_track_execution(self):
        graph = build_triage_graph()
        with (
            patch("ai_triage_agent.agents.log_analyzer.call_llm_json", return_value=_MOCK_LOG_ANALYSIS),
            patch("ai_triage_agent.agents.diff_analyzer.call_llm_json", return_value=_MOCK_DIFF_ANALYSIS),
            patch("ai_triage_agent.agents.rca_synthesizer.call_llm_json", return_value=_MOCK_RCA),
        ):
            result = graph.invoke({
                "run_id": "test_run_003",
                "raw_log": "some log",
                "git_diff": "some diff",
                "errors": [],
                "completed_nodes": [],
            })

        assert set(result["completed_nodes"]) == {"log_analyzer", "diff_analyzer", "rca_synthesizer"}
