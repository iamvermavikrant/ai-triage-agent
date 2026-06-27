"""DeepEval-backed pytest tests for RCA quality.

These tests run in mock mode by default (MOCK_LLM=true).
Set MOCK_LLM=false and provide ANTHROPIC_API_KEY + OPENAI_API_KEY
(or configure deepeval to use Anthropic) to run real evaluations.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from evals.deepeval_metrics import run_deepeval, _MOCK_DEEPEVAL_RESULT

# ── Sample RCA outputs for testing ────────────────────────────────────────────

_GOOD_RCA = {
    "title": "CUDA OOM: BATCH_SIZE increased 8 to 64 without VRAM budget analysis",
    "root_cause": (
        "trainer.py changed BATCH_SIZE from 8 to 64, requiring ~14 GiB GPU memory. "
        "The target GPU has ~12 GiB free after model loading, causing OOM in _load_batch."
    ),
    "contributing_factors": ["No VRAM budget gate in CI"],
    "blast_radius": "All training tests",
    "recommended_fix": "Revert BATCH_SIZE to 8 or add gradient accumulation (accumulation_steps=8).",
    "preventive_measures": ["Add CI VRAM profiling step"],
    "priority": "P1",
    "estimated_fix_time": "2h",
    "owner_hint": "ml-training team",
}

_BAD_RCA = {
    "title": "Some error occurred",
    "root_cause": "The system encountered an unexpected issue.",
    "contributing_factors": [],
    "blast_radius": "Unknown",
    "recommended_fix": "Investigate the problem and fix it.",
    "preventive_measures": [],
    "priority": "P3",
    "estimated_fix_time": "unknown",
    "owner_hint": "unknown team",
}

_GROUND_TRUTH = {
    "failure_type": "CUDA_OOM",
    "root_cause": "Batch size increase from 8 to 64 caused GPU OOM.",
    "priority": "P1",
    "recommended_fix": "Revert BATCH_SIZE or implement gradient accumulation.",
    "implicated_file": "src/training/trainer.py",
}

_SAMPLE_LOG = "ERROR: CUDA out of memory. Tried to allocate 14.00 GiB"
_SAMPLE_DIFF = "- BATCH_SIZE = 8\n+ BATCH_SIZE = 64"


class TestDeepEvalMockMode:
    """Tests that run without any API key using mock responses."""

    def test_mock_mode_returns_result(self):
        result = run_deepeval(_GOOD_RCA, _GROUND_TRUTH, _SAMPLE_LOG, _SAMPLE_DIFF)
        assert "geval" in result
        assert "hallucination" in result
        assert "answer_relevancy" in result
        assert "deepeval_overall_pass" in result
        assert "deepeval_summary" in result

    def test_mock_mode_geval_has_three_criteria(self):
        result = run_deepeval(_GOOD_RCA, _GROUND_TRUTH)
        geval = result["geval"]
        assert "rca_correctness" in geval
        assert "fix_actionability" in geval
        assert "no_scope_creep" in geval

    def test_mock_mode_scores_in_range(self):
        result = run_deepeval(_GOOD_RCA, _GROUND_TRUTH)
        for key, val in result["geval"].items():
            assert 0.0 <= val["score"] <= 1.0, f"{key} score out of range"
        assert 0.0 <= result["hallucination"]["score"] <= 1.0
        assert 0.0 <= result["answer_relevancy"]["score"] <= 1.0

    def test_mock_mode_good_rca_passes(self):
        result = run_deepeval(_GOOD_RCA, _GROUND_TRUTH, _SAMPLE_LOG, _SAMPLE_DIFF)
        assert result["deepeval_overall_pass"] is True

    def test_mock_hallucination_score_below_threshold(self):
        result = run_deepeval(_GOOD_RCA, _GROUND_TRUTH)
        # Hallucination score = hallucination rate; should be < 0.3 to pass
        assert result["hallucination"]["score"] < 0.3
        assert result["hallucination"]["passed"] is True

    def test_mock_answer_relevancy_above_threshold(self):
        result = run_deepeval(_GOOD_RCA, _GROUND_TRUTH)
        assert result["answer_relevancy"]["score"] >= 0.7
        assert result["answer_relevancy"]["passed"] is True


class TestDeepEvalNotInstalled:
    """Graceful degradation when deepeval package is missing."""

    def test_falls_back_to_mock_when_not_installed(self, monkeypatch):
        monkeypatch.setenv("MOCK_LLM", "false")
        with patch.dict("sys.modules", {"deepeval": None}):
            result = run_deepeval(_GOOD_RCA, _GROUND_TRUTH)
        # Should return mock scores, not raise
        assert "geval" in result
        assert "not installed" in result["deepeval_summary"].lower()


class TestDeepEvalIntegration:
    """
    Integration tests that call real DeepEval APIs.

    Skipped unless MOCK_LLM=false AND ANTHROPIC_API_KEY is set.
    Run with: pytest tests/test_deepeval.py::TestDeepEvalIntegration -v
    """

    @pytest.fixture(autouse=True)
    def require_real_llm(self, monkeypatch):
        if os.getenv("MOCK_LLM", "true").lower() == "true":
            pytest.skip("Set MOCK_LLM=false to run real DeepEval tests")
        try:
            import deepeval  # noqa: F401
        except ImportError:
            pytest.skip("deepeval not installed — pip install deepeval")

    def test_good_rca_passes_all_metrics(self):
        result = run_deepeval(_GOOD_RCA, _GROUND_TRUTH, _SAMPLE_LOG, _SAMPLE_DIFF)
        assert result["deepeval_overall_pass"] is True, result["deepeval_summary"]

    def test_bad_rca_fails_geval_correctness(self):
        result = run_deepeval(_BAD_RCA, _GROUND_TRUTH, _SAMPLE_LOG, _SAMPLE_DIFF)
        correctness = result["geval"]["rca_correctness"]
        assert correctness["score"] < 0.7, (
            f"Expected bad RCA to fail correctness, got score={correctness['score']}"
        )

    def test_bad_rca_fails_actionability(self):
        result = run_deepeval(_BAD_RCA, _GROUND_TRUTH, _SAMPLE_LOG, _SAMPLE_DIFF)
        actionability = result["geval"]["fix_actionability"]
        assert actionability["score"] < 0.7, (
            f"Expected vague fix to fail actionability, got score={actionability['score']}"
        )
