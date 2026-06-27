"""Thin wrapper around the Anthropic SDK with retry and structured output.

Set MOCK_LLM=true in your environment to run the full pipeline without an
API key — each agent returns a realistic hardcoded response based on which
prompt it receives.
"""

from __future__ import annotations

import json
import os
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)


# ── Mock responses (used when MOCK_LLM=true) ──────────────────────────────────

_MOCK_LOG_ANALYSIS_BY_FIXTURE: dict[str, dict[str, Any]] = {
    "CUDA_OOM": {
        "failure_type": "CUDA_OOM",
        "error_summary": "GPU ran out of memory during batch loading after BATCH_SIZE was increased from 8 to 64.",
        "stack_frames": ["src/training/trainer.py:312 in _load_batch", "tests/test_model_training.py:88"],
        "affected_modules": ["src/training/trainer.py"],
        "severity": "CRITICAL",
        "reproducible": True,
        "keywords": ["OOM", "CUDA", "RuntimeError", "batch", "14 GiB"],
    },
    "IMPORT_ERROR": {
        "failure_type": "IMPORT_ERROR",
        "error_summary": "Cannot import FlashAttentionKernel from nvidia_attention — package or symbol does not exist.",
        "stack_frames": ["src/models/attention.py:5 in <module>", "tests/test_inference.py:3 in <module>"],
        "affected_modules": ["src/models/attention.py"],
        "severity": "CRITICAL",
        "reproducible": True,
        "keywords": ["ImportError", "FlashAttentionKernel", "nvidia_attention", "collection"],
    },
    "ASSERTION_FAILED": {
        "failure_type": "ASSERTION_FAILED",
        "error_summary": "Throughput regression of 38.7% detected — current 871 tokens/sec vs baseline 1420 tokens/sec.",
        "stack_frames": ["tests/test_benchmarks.py:67 in test_benchmark_throughput"],
        "affected_modules": ["src/attention/scaled_dot_product.py", "src/training/trainer.py"],
        "severity": "HIGH",
        "reproducible": True,
        "keywords": ["throughput", "regression", "AssertionError", "tokens/sec", "baseline"],
    },
    "TIMEOUT": {
        "failure_type": "TIMEOUT",
        "error_summary": "NCCL allreduce barrier timed out on rank 2 after 1800s during distributed training.",
        "stack_frames": ["src/distributed/nccl_utils.py:144 in _allreduce_with_timeout", "tests/test_distributed.py:55"],
        "affected_modules": ["src/distributed/nccl_utils.py"],
        "severity": "HIGH",
        "reproducible": False,
        "keywords": ["TimeoutError", "NCCL", "allreduce", "rank 2", "barrier", "intermittent"],
    },
    "ENV_MISMATCH": {
        "failure_type": "ENV_MISMATCH",
        "error_summary": "CUDA version mismatch: CI has 11.8 but code requires >= 12.1 for BF16 tensor core support.",
        "stack_frames": ["src/utils/cuda_check.py:28 in assert_cuda_version", "tests/test_compatibility.py:19"],
        "affected_modules": ["src/utils/cuda_check.py"],
        "severity": "CRITICAL",
        "reproducible": True,
        "keywords": ["AssertionError", "CUDA", "11.8", "12.1", "BF16", "environment"],
    },
}

_MOCK_LOG_ANALYSIS = _MOCK_LOG_ANALYSIS_BY_FIXTURE["CUDA_OOM"]

_MOCK_DIFF_ANALYSIS = {
    "implicated_files": [
        {
            "file": "src/training/trainer.py",
            "hunk_summary": "BATCH_SIZE constant changed from 8 to 64 (8x increase)",
            "relevance_score": 0.97,
            "reasoning": "The 8x batch size increase directly explains the 14 GiB allocation failure on a GPU with ~12 GiB free.",
        }
    ],
    "change_risk": "HIGH",
    "regression_likely": True,
    "confidence": 0.95,
}

_MOCK_RCA = {
    "title": "CUDA OOM: BATCH_SIZE increased 8 to 64 without VRAM budget analysis",
    "root_cause": (
        "trainer.py changed BATCH_SIZE from 8 to 64, requiring ~14 GiB of GPU memory per batch. "
        "The target GPU has only ~12 GiB free after model weights are loaded, causing an OOM on the "
        "first _load_batch call in test_full_training_run."
    ),
    "contributing_factors": [
        "No VRAM budget gate in CI to catch memory-intensive changes",
        "pin_memory() added alongside the batch size increase, adding extra host memory pressure",
    ],
    "blast_radius": "All training tests; any job using this trainer on <=24 GiB GPUs",
    "recommended_fix": (
        "Revert BATCH_SIZE to 8 in trainer.py, or implement gradient accumulation "
        "(e.g. accumulation_steps=8) to simulate a larger effective batch without exceeding VRAM."
    ),
    "preventive_measures": [
        "Add a CI step that profiles peak VRAM usage and fails if it exceeds 80% of GPU capacity",
        "Document VRAM requirements in the trainer class docstring",
        "Add a unit test that asserts BATCH_SIZE <= safe limit for the test GPU tier",
    ],
    "priority": "P1",
    "estimated_fix_time": "2h",
    "owner_hint": "ml-training team",
}

# Per-fixture judge responses — keyed by failure_type in the user prompt
_MOCK_JUDGE_BY_FIXTURE: dict[str, dict[str, Any]] = {
    "CUDA_OOM": {
        "scores": {"accuracy": 9, "completeness": 9, "actionability": 9, "precision": 10, "clarity": 9},
        "weighted_total": 9.1,
        "pass": True,
        "critique": (
            "Root cause precisely identifies the 8x batch size increase as the OOM trigger. "
            "Fix is immediately actionable (gradient accumulation). "
            "Preventive measures are enforceable in CI."
        ),
    },
    "IMPORT_ERROR": {
        "scores": {"accuracy": 10, "completeness": 9, "actionability": 10, "precision": 9, "clarity": 10},
        "weighted_total": 9.6,
        "pass": True,
        "critique": (
            "Perfect identification of the renamed import symbol as root cause. "
            "One-line fix is crystal clear. "
            "Minor: does not suggest adding an import smoke test to CI."
        ),
    },
    "ASSERTION_FAILED": {
        "scores": {"accuracy": 8, "completeness": 8, "actionability": 8, "precision": 9, "clarity": 8},
        "weighted_total": 8.2,
        "pass": True,
        "critique": (
            "Correctly flags missing scale factor and LR multiplier. "
            "Fix is actionable but could specify the exact line to change. "
            "Blast radius assessment is slightly conservative."
        ),
    },
    "TIMEOUT": {
        "scores": {"accuracy": 7, "completeness": 6, "actionability": 6, "precision": 8, "clarity": 7},
        "weighted_total": 6.7,
        "pass": False,
        "critique": (
            "Intermittent nature correctly identified. "
            "Root cause is speculative — thermal throttling not confirmed. "
            "Fix lacks specificity: 'add retry logic' is too vague for a P2 incident."
        ),
    },
    "ENV_MISMATCH": {
        "scores": {"accuracy": 9, "completeness": 9, "actionability": 9, "precision": 10, "clarity": 9},
        "weighted_total": 9.1,
        "pass": True,
        "critique": (
            "Correctly attributes failure to CUDA 11.8 vs 12.1 requirement mismatch. "
            "Both fix paths (update Docker image or runtime capability check) are actionable. "
            "Excellent preventive measure suggestion."
        ),
    },
}

# Fallback for unrecognised fixture
_MOCK_JUDGE = _MOCK_JUDGE_BY_FIXTURE["CUDA_OOM"]

# Per-fixture RCA responses
_MOCK_RCA_BY_FIXTURE: dict[str, dict[str, Any]] = {
    "CUDA_OOM": {
        "title": "CUDA OOM: BATCH_SIZE increased 8 to 64 without VRAM budget analysis",
        "root_cause": (
            "trainer.py changed BATCH_SIZE from 8 to 64, requiring ~14 GiB GPU memory. "
            "The target GPU has ~12 GiB free after model loading, causing OOM in _load_batch."
        ),
        "contributing_factors": ["No VRAM budget gate in CI", "pin_memory() added simultaneously"],
        "blast_radius": "All training tests on GPUs with less than 24 GiB VRAM",
        "recommended_fix": "Revert BATCH_SIZE to 8 or add gradient accumulation (accumulation_steps=8).",
        "preventive_measures": ["Add CI VRAM profiling step", "Document VRAM requirements in trainer"],
        "priority": "P1",
        "estimated_fix_time": "2h",
        "owner_hint": "ml-training team",
    },
    "IMPORT_ERROR": {
        "title": "P0 ImportError: FlashAttentionKernel not found — all inference tests blocked",
        "root_cause": (
            "attention.py import was renamed from 'nvidia_flash_attention.FlashAttention' "
            "to 'nvidia_attention.FlashAttentionKernel' which does not exist, "
            "blocking collection of the entire test_inference suite."
        ),
        "contributing_factors": ["No import smoke test in CI", "Package rename not verified before merge"],
        "blast_radius": "All inference tests — full test suite collection blocked",
        "recommended_fix": "Revert import in src/models/attention.py to 'from nvidia_flash_attention import FlashAttention'.",
        "preventive_measures": ["Add import validation test", "Pin package names in requirements.txt"],
        "priority": "P0",
        "estimated_fix_time": "30m",
        "owner_hint": "ml-inference team",
    },
    "ASSERTION_FAILED": {
        "title": "38.7% Throughput Regression: Missing scale factor in attention + 10x LR increase",
        "root_cause": (
            "ScaledDotProductAttention einsum refactor dropped the 1/sqrt(d_k) scaling factor, "
            "causing attention score explosion. Compounded by a 10x learning rate multiplier "
            "in configure_optimizers destabilising training convergence."
        ),
        "contributing_factors": ["No throughput regression gate in CI", "Two high-risk changes in one PR"],
        "blast_radius": "All benchmark tests; model quality degraded on all downstream tasks",
        "recommended_fix": "Restore scale: divide einsum by math.sqrt(d_k). Revert LR multiplier in configure_optimizers.",
        "preventive_measures": ["Add throughput regression CI gate (threshold: -5%)", "Require perf sign-off for attention changes"],
        "priority": "P1",
        "estimated_fix_time": "3h",
        "owner_hint": "ml-architecture team",
    },
    "TIMEOUT": {
        "title": "Intermittent NCCL Timeout: Rank 2 missing allreduce barrier (3/20 runs)",
        "root_cause": (
            "NCCL allreduce barrier on rank 2 timed out after 1800s during distributed training. "
            "Failure is intermittent (15% rate), suggesting node-level issues such as GPU thermal "
            "throttling or NVLink bandwidth saturation rather than a code regression."
        ),
        "contributing_factors": ["No per-rank timing instrumentation", "No NVLink health check in CI"],
        "blast_radius": "Distributed training tests on affected node; other nodes unaffected",
        "recommended_fix": "Add NCCL_DEBUG=INFO, instrument per-rank timing, pin test to healthy node.",
        "preventive_measures": ["Add NVLink health check pre-test", "Alert on >10% flake rate for distributed tests"],
        "priority": "P2",
        "estimated_fix_time": "1d",
        "owner_hint": "infra / distributed-systems team",
    },
    "ENV_MISMATCH": {
        "title": "CUDA Version Mismatch: CI has 11.8, code now requires 12.1 for BF16",
        "root_cause": (
            "cuda_check.py assertion was tightened to require CUDA >= 12.1 for BF16 tensor core support, "
            "but the CI runner Docker image still has CUDA 11.8. "
            "The check fires immediately at environment validation before any tests run."
        ),
        "contributing_factors": ["CI Docker image not updated alongside code requirement bump", "No environment matrix in CI"],
        "blast_radius": "All tests on CI — environment validation fails before any test executes",
        "recommended_fix": "Update CI Docker image to nvidia/cuda:12.1.0-devel-ubuntu22.04, or gate BF16 path behind torch.cuda.is_bf16_supported().",
        "preventive_measures": ["Pin CUDA version in CI matrix", "Add environment compatibility test as first CI step"],
        "priority": "P1",
        "estimated_fix_time": "4h",
        "owner_hint": "devops / ml-platform team",
    },
}


def _mock_response(system: str, user: str = "") -> dict[str, Any]:
    """Pick the right mock payload based on which agent is calling.

    Order matters: judge check must come before rca_synthesizer because
    the judge prompt also contains the word 'RCA'.

    For judge and rca_synthesizer, we key off the user prompt content
    so each fixture gets a distinct, realistic response.
    """
    s = system.lower()
    u = user.lower()

    def _fixture_key() -> str:
        # Check most specific signals first to avoid collisions
        if "importerror" in u or "flashattentionkernel" in u or "import_error" in u:
            return "IMPORT_ERROR"
        if "throughput" in u or "einsum" in u or "tokens/sec" in u or "assertion_failed" in u:
            return "ASSERTION_FAILED"
        if "nccl" in u or "allreduce" in u or "rank 2" in u or "timeout" in u and "nccl" in u:
            return "TIMEOUT"
        if "11.8" in u or "bf16" in u or "env_mismatch" in u or "cuda version mismatch" in u:
            return "ENV_MISMATCH"
        if "cuda out of memory" in u or "cuda_oom" in u or "batch_size" in u or "14 gib" in u:
            return "CUDA_OOM"
        return "CUDA_OOM"  # default

    if "impartial evaluator" in s:
        return _MOCK_JUDGE_BY_FIXTURE.get(_fixture_key(), _MOCK_JUDGE)
    if "log analyst" in s or "failure signals" in s:
        return _MOCK_LOG_ANALYSIS_BY_FIXTURE.get(_fixture_key(), _MOCK_LOG_ANALYSIS)
    if "change-impact" in s or "git diff" in s:
        return _MOCK_DIFF_ANALYSIS
    if "root cause analysis" in s or "rca synthesizer" in s or "principal sdet" in s:
        return _MOCK_RCA_BY_FIXTURE.get(_fixture_key(), _MOCK_RCA)
    return {"mock": True, "note": "unrecognised agent — returning empty mock"}


# ── Real client (lazy-loaded only when MOCK_LLM != true) ──────────────────────

def _get_client():  # type: ignore[return]
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Set MOCK_LLM=true in your .env to run without a real key."
        )
    return Anthropic(api_key=api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(
    *,
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    """Call the Anthropic API and return the raw text response."""
    if os.getenv("MOCK_LLM", "false").lower() == "true":
        payload = _mock_response(system, user)
        log.info("llm_call.mock", agent_hint=list(payload.keys())[:2])
        return json.dumps(payload)

    resolved_model = model or os.getenv("TRIAGE_MODEL", "claude-sonnet-4-6")
    log.info("llm_call", model=resolved_model, user_len=len(user))
    client = _get_client()
    message = client.messages.create(
        model=resolved_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text  # type: ignore[no-any-return]


def call_llm_json(
    *,
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Call the LLM and parse the response as JSON."""
    raw = call_llm(system=system, user=user, model=model, max_tokens=max_tokens)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    return json.loads(cleaned)  # type: ignore[no-any-return]
