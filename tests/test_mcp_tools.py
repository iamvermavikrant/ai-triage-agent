"""Tests for MCP tools: fetch_test_logs and get_git_diff."""

from __future__ import annotations

import pytest

from ai_triage_agent.mcp.tools.fetch_test_logs import fetch_test_logs
from ai_triage_agent.mcp.tools.get_git_diff import get_git_diff


class TestFetchTestLogs:
    def test_mock_cuda_oom(self):
        log = fetch_test_logs(run_id="cuda_oom", backend="mock")
        assert "CUDA out of memory" in log
        assert "RuntimeError" in log

    def test_mock_import_error(self):
        log = fetch_test_logs(run_id="import_error", backend="mock")
        assert "ImportError" in log
        assert "FlashAttentionKernel" in log

    def test_mock_flaky_timeout(self):
        log = fetch_test_logs(run_id="flaky_timeout", backend="mock")
        assert "TimeoutError" in log
        assert "NCCL" in log

    def test_mock_env_mismatch(self):
        log = fetch_test_logs(run_id="env_mismatch", backend="mock")
        assert "CUDA version mismatch" in log
        assert "11.8" in log

    def test_unknown_run_id_returns_placeholder(self):
        log = fetch_test_logs(run_id="nonexistent_run_999", backend="mock")
        assert "no mock log" in log

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            fetch_test_logs(run_id="cuda_oom", backend="s3")

    def test_fixture_prefix_stripped(self):
        log = fetch_test_logs(run_id="fixture_01_cuda_oom", backend="mock")
        assert "CUDA out of memory" in log


class TestGetGitDiff:
    def test_mock_regression_diff(self):
        diff = get_git_diff(commit_sha="regression_diff", backend="mock")
        assert "scaled_dot_product.py" in diff
        assert "einsum" in diff

    def test_mock_cuda_oom_diff(self):
        diff = get_git_diff(commit_sha="cuda_oom", backend="mock")
        assert "BATCH_SIZE" in diff

    def test_mock_import_error_diff(self):
        diff = get_git_diff(commit_sha="import_error", backend="mock")
        assert "FlashAttentionKernel" in diff

    def test_flaky_timeout_no_diff(self):
        diff = get_git_diff(commit_sha="flaky_timeout", backend="mock")
        assert diff == ""

    def test_fixture_key_prefix_handling(self):
        diff = get_git_diff(commit_sha="fixture_03_regression_diff", backend="mock")
        assert "einsum" in diff
