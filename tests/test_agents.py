"""Unit tests for individual agent nodes using mocked LLM calls."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_triage_agent.agents.diff_analyzer import diff_analyzer_node
from ai_triage_agent.agents.log_analyzer import log_analyzer_node
from ai_triage_agent.agents.rca_synthesizer import rca_synthesizer_node


_MOCK_LOG_ANALYSIS = {
    "failure_type": "CUDA_OOM",
    "error_summary": "GPU ran out of memory during batch loading.",
    "stack_frames": ["trainer.py:312", "trainer.py:289"],
    "affected_modules": ["src/training/trainer.py"],
    "severity": "CRITICAL",
    "reproducible": True,
    "keywords": ["OOM", "CUDA", "batch"],
}

_MOCK_DIFF_ANALYSIS = {
    "implicated_files": [
        {
            "file": "src/training/trainer.py",
            "hunk_summary": "BATCH_SIZE changed from 8 to 64",
            "relevance_score": 0.97,
            "reasoning": "8x batch size increase directly explains the OOM.",
        }
    ],
    "change_risk": "HIGH",
    "regression_likely": True,
    "confidence": 0.95,
}

_MOCK_RCA = {
    "title": "CUDA OOM: batch size 8→64 exceeds GPU VRAM",
    "root_cause": "Batch size was increased 8x without memory budget analysis.",
    "contributing_factors": ["No gradient checkpointing", "No memory profiling in CI"],
    "blast_radius": "All training tests",
    "recommended_fix": "Revert BATCH_SIZE to 8 or add gradient accumulation.",
    "preventive_measures": ["Add VRAM budget CI check", "Enforce OOM tests"],
    "priority": "P1",
    "estimated_fix_time": "2h",
    "owner_hint": "ml-training team",
}


class TestLogAnalyzerNode:
    def test_returns_log_analysis(self):
        state = {"run_id": "t01", "raw_log": "ERROR: CUDA OOM"}
        with patch("ai_triage_agent.agents.log_analyzer.call_llm_json", return_value=_MOCK_LOG_ANALYSIS):
            result = log_analyzer_node(state)
        assert result["log_analysis"]["failure_type"] == "CUDA_OOM"
        assert "log_analyzer" in result["completed_nodes"]

    def test_empty_log_adds_error(self):
        state = {"run_id": "t02", "raw_log": ""}
        result = log_analyzer_node(state)
        assert result.get("errors")
        assert "log_analyzer" in result["errors"][0]

    def test_missing_raw_log_key(self):
        state = {"run_id": "t03"}
        result = log_analyzer_node(state)
        assert result.get("errors")


class TestDiffAnalyzerNode:
    def test_returns_diff_analysis(self):
        state = {
            "run_id": "t04",
            "log_analysis": _MOCK_LOG_ANALYSIS,
            "git_diff": "diff --git a/trainer.py ...",
        }
        with patch("ai_triage_agent.agents.diff_analyzer.call_llm_json", return_value=_MOCK_DIFF_ANALYSIS):
            result = diff_analyzer_node(state)
        assert result["diff_analysis"]["change_risk"] == "HIGH"
        assert "diff_analyzer" in result["completed_nodes"]

    def test_no_diff_returns_low_risk_placeholder(self):
        state = {
            "run_id": "t05",
            "log_analysis": _MOCK_LOG_ANALYSIS,
            "git_diff": "",
        }
        result = diff_analyzer_node(state)
        assert result["diff_analysis"]["change_risk"] == "LOW"
        assert result["diff_analysis"]["confidence"] == 0.0

    def test_missing_log_analysis_adds_error(self):
        state = {"run_id": "t06", "git_diff": "diff..."}
        result = diff_analyzer_node(state)
        assert result.get("errors")


class TestRCASynthesizerNode:
    def test_returns_rca_report(self):
        state = {
            "run_id": "t07",
            "log_analysis": _MOCK_LOG_ANALYSIS,
            "diff_analysis": _MOCK_DIFF_ANALYSIS,
            "test_suite": "test_training",
            "commit_sha": "abc123",
            "branch": "main",
        }
        with patch("ai_triage_agent.agents.rca_synthesizer.call_llm_json", return_value=_MOCK_RCA):
            result = rca_synthesizer_node(state)
        assert result["rca_report"]["priority"] == "P1"
        assert "rca_synthesizer" in result["completed_nodes"]

    def test_missing_inputs_adds_error(self):
        state = {"run_id": "t08"}
        result = rca_synthesizer_node(state)
        assert result.get("errors")
