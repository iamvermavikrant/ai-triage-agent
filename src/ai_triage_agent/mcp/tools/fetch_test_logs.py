"""MCP tool: fetch_test_logs — retrieves CI test logs by run ID."""

from __future__ import annotations

import os
import re
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

# Supported backends
_BACKENDS = {"github_actions", "local_file", "mock"}


def fetch_test_logs(run_id: str, backend: str = "github_actions", **kwargs: Any) -> str:
    """
    Fetch test logs for a given CI run ID.

    Args:
        run_id: GitHub Actions run ID, local file path, or fixture key.
        backend: One of 'github_actions', 'local_file', 'mock'.

    Returns:
        Raw log text.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {_BACKENDS}")

    log.info("fetch_test_logs", run_id=run_id, backend=backend)

    if backend == "github_actions":
        return _fetch_github_actions_log(run_id)
    if backend == "local_file":
        return _fetch_local_file(run_id)
    return _fetch_mock_log(run_id)


# ── Backend implementations ────────────────────────────────────────────────────

def _fetch_github_actions_log(run_id: str) -> str:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPO", "owner/repo")

    if not token:
        log.warning("fetch_test_logs.no_token", hint="falling back to mock")
        return _fetch_mock_log(run_id)

    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/logs"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        # GitHub returns a zip; extract first .txt for simplicity
        return _extract_zip_text(resp.content)


def _extract_zip_text(content: bytes) -> str:
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
        if not txt_files:
            return "<no log files found in zip>"
        # Return the largest log file (usually the test runner output)
        largest = max(txt_files, key=lambda n: zf.getinfo(n).file_size)
        return zf.read(largest).decode("utf-8", errors="replace")


def _fetch_local_file(path: str) -> str:
    with open(path) as fh:
        return fh.read()


def _fetch_mock_log(run_id: str) -> str:
    """Returns canned logs keyed by well-known run IDs for demo/eval purposes."""
    mocks: dict[str, str] = {
        "cuda_oom": _CUDA_OOM_LOG,
        "import_error": _IMPORT_ERROR_LOG,
        "regression_diff": _REGRESSION_LOG,
        "flaky_timeout": _FLAKY_TIMEOUT_LOG,
        "env_mismatch": _ENV_MISMATCH_LOG,
    }
    # Strip numeric suffixes so fixture IDs like "fixture_01_cuda_oom" work too
    key = re.sub(r"^fixture_\d+_", "", run_id)
    return mocks.get(key, f"<no mock log for run_id='{run_id}'>")


# ── Canned mock logs ───────────────────────────────────────────────────────────

_CUDA_OOM_LOG = """\
2024-06-27T10:15:03Z [INFO] Starting test suite: test_model_training
2024-06-27T10:15:04Z [INFO] Loading model weights from /models/llama-70b
2024-06-27T10:15:45Z [INFO] Moving model to CUDA device cuda:0
2024-06-27T10:15:46Z [ERROR] RuntimeError: CUDA out of memory. Tried to allocate 14.00 GiB
  (GPU 0; 79.20 GiB total capacity; 64.32 GiB already allocated; 12.89 GiB free;
   65.11 GiB reserved in total by PyTorch)
  File "src/training/trainer.py", line 312, in _load_batch
    batch = batch.to(self.device)
  File "src/training/trainer.py", line 289, in train_epoch
    loss = self.model(batch)
  File "tests/test_model_training.py", line 88, in test_full_training_run
    trainer.train(epochs=1)
2024-06-27T10:15:46Z [CRITICAL] Test FAILED: test_model_training.test_full_training_run
2024-06-27T10:15:46Z [INFO] 0 passed, 1 failed, 0 errors in 42.3s
"""

_IMPORT_ERROR_LOG = """\
2024-06-27T09:00:01Z [INFO] Collecting tests from tests/test_inference.py
2024-06-27T09:00:02Z [ERROR] ImportError: cannot import name 'FlashAttentionKernel' from 'nvidia_attention' (unknown location)
  File "src/models/attention.py", line 5, in <module>
    from nvidia_attention import FlashAttentionKernel
  File "tests/test_inference.py", line 3, in <module>
    from src.models.attention import MultiHeadAttention
2024-06-27T09:00:02Z [CRITICAL] ERROR collecting tests/test_inference.py
2024-06-27T09:00:02Z [INFO] 0 passed, 0 failed, 1 error in 1.1s
"""

_REGRESSION_LOG = """\
2024-06-27T11:30:00Z [INFO] Running test_benchmark_throughput
2024-06-27T11:30:05Z [INFO] Baseline throughput: 1420 tokens/sec
2024-06-27T11:30:10Z [INFO] Current throughput: 871 tokens/sec
2024-06-27T11:30:10Z [FAIL] AssertionError: Throughput regression detected
  Expected >= 1350 tokens/sec, got 871 tokens/sec (38.7% drop)
  File "tests/test_benchmarks.py", line 67, in test_benchmark_throughput
    assert throughput >= BASELINE_THRESHOLD, f"Throughput regression..."
2024-06-27T11:30:10Z [CRITICAL] Test FAILED: test_benchmarks.test_benchmark_throughput
"""

_FLAKY_TIMEOUT_LOG = """\
2024-06-27T08:45:00Z [INFO] Running test_distributed_training (4 GPUs)
2024-06-27T08:45:01Z [INFO] Initializing NCCL process group
2024-06-27T09:15:01Z [ERROR] TimeoutError: NCCL collective operation timed out after 1800s
  Rank 2 did not respond during allreduce barrier
  File "src/distributed/nccl_utils.py", line 144, in _allreduce_with_timeout
    dist.barrier(timeout=datetime.timedelta(seconds=1800))
  File "tests/test_distributed.py", line 55, in test_distributed_training
    engine.run(epochs=2)
2024-06-27T09:15:01Z [WARNING] This failure is intermittent (3rd occurrence in 20 runs)
2024-06-27T09:15:01Z [CRITICAL] Test FAILED: test_distributed.test_distributed_training
"""

_ENV_MISMATCH_LOG = """\
2024-06-27T07:00:00Z [INFO] Running test_cuda_version_compatibility
2024-06-27T07:00:01Z [INFO] Detected CUDA 11.8 driver
2024-06-27T07:00:01Z [ERROR] AssertionError: CUDA version mismatch
  Required: CUDA >= 12.1 (for BF16 tensor core support)
  Found: CUDA 11.8
  File "src/utils/cuda_check.py", line 28, in assert_cuda_version
    assert cuda_major >= 12 and cuda_minor >= 1
  File "tests/test_compatibility.py", line 19, in test_cuda_version_compatibility
    assert_cuda_version()
2024-06-27T07:00:01Z [CRITICAL] Test FAILED: environment does not meet minimum requirements
"""
