"""MCP tool: get_git_diff — retrieves a unified diff for a commit or PR."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)


def get_git_diff(
    commit_sha: str,
    repo_path: str | None = None,
    backend: str = "github_api",
    **kwargs: Any,
) -> str:
    """
    Retrieve the unified git diff for a given commit SHA.

    Args:
        commit_sha: Full or short commit SHA, or 'HEAD', or 'fixture_<name>'.
        repo_path: Local path to a git repo (used when backend='local').
        backend: One of 'github_api', 'local', 'mock'.

    Returns:
        Unified diff as a string.
    """
    log.info("get_git_diff", commit_sha=commit_sha, backend=backend)

    if commit_sha.startswith("fixture_") or backend == "mock":
        return _mock_diff(commit_sha)

    if backend == "local":
        return _local_diff(commit_sha, repo_path)

    return _github_api_diff(commit_sha)


# ── Backend implementations ────────────────────────────────────────────────────

def _github_api_diff(commit_sha: str) -> str:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPO", "owner/repo")

    if not token:
        log.warning("get_git_diff.no_token", hint="returning mock diff")
        return _mock_diff(commit_sha)

    url = f"https://api.github.com/repos/{repo}/commits/{commit_sha}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text


def _local_diff(commit_sha: str, repo_path: str | None) -> str:
    cwd = Path(repo_path) if repo_path else Path.cwd()
    result = subprocess.run(
        ["git", "diff", f"{commit_sha}^", commit_sha],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _mock_diff(key: str) -> str:
    mocks: dict[str, str] = {
        "regression_diff": _REGRESSION_DIFF,
        "cuda_oom": _CUDA_OOM_DIFF,
        "import_error": _IMPORT_ERROR_DIFF,
        "flaky_timeout": "",
        "env_mismatch": _ENV_MISMATCH_DIFF,
    }
    clean_key = key.replace("fixture_03_", "").replace("fixture_01_", "").replace("fixture_", "")
    return mocks.get(clean_key, mocks.get(key, ""))


# ── Canned diffs ───────────────────────────────────────────────────────────────

_REGRESSION_DIFF = """\
diff --git a/src/attention/scaled_dot_product.py b/src/attention/scaled_dot_product.py
index a3f1c2b..8e4d9a1 100644
--- a/src/attention/scaled_dot_product.py
+++ b/src/attention/scaled_dot_product.py
@@ -41,7 +41,10 @@ class ScaledDotProductAttention(nn.Module):
     def forward(self, q, k, v, mask=None):
         d_k = q.size(-1)
-        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
+        # Switched to einsum for clarity — NOTE: missing scale factor
+        scores = torch.einsum('bhid,bhjd->bhij', q, k)
         if mask is not None:
             scores = scores.masked_fill(mask == 0, -1e9)
         attn = F.softmax(scores, dim=-1)
diff --git a/src/training/trainer.py b/src/training/trainer.py
index 2c8f901..1a2d347 100644
--- a/src/training/trainer.py
+++ b/src/training/trainer.py
@@ -78,6 +78,8 @@ class Trainer:
     def configure_optimizers(self):
-        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
+        # Increased learning rate for faster convergence
+        return torch.optim.AdamW(self.model.parameters(), lr=self.lr * 10)
"""

_CUDA_OOM_DIFF = """\
diff --git a/src/training/trainer.py b/src/training/trainer.py
index 1b2c3d4..5e6f7a8 100644
--- a/src/training/trainer.py
+++ b/src/training/trainer.py
@@ -305,6 +305,7 @@ class Trainer:
     def _load_batch(self, batch):
+        batch = batch.pin_memory()
         return batch.to(self.device)

@@ -280,7 +280,7 @@ class Trainer:
-    BATCH_SIZE = 8
+    BATCH_SIZE = 64
"""

_IMPORT_ERROR_DIFF = """\
diff --git a/src/models/attention.py b/src/models/attention.py
index abc1234..def5678 100644
--- a/src/models/attention.py
+++ b/src/models/attention.py
@@ -1,6 +1,6 @@
 import torch
 import torch.nn as nn
-from nvidia_flash_attention import FlashAttention
+from nvidia_attention import FlashAttentionKernel
 from typing import Optional
"""

_ENV_MISMATCH_DIFF = """\
diff --git a/pyproject.toml b/pyproject.toml
index 111aaaa..222bbbb 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -12,6 +12,6 @@ dependencies = [
-    "torch>=2.0.0",
+    "torch>=2.3.0",
+    "triton>=2.3.0",
 ]
diff --git a/src/utils/cuda_check.py b/src/utils/cuda_check.py
@@ -25,5 +25,5 @@ def assert_cuda_version():
-    assert cuda_major >= 11
+    assert cuda_major >= 12 and cuda_minor >= 1, "BF16 tensor core requires CUDA 12.1+"
"""
