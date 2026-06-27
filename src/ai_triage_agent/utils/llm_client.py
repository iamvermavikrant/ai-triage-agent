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

_MOCK_LOG_ANALYSIS = {
    "failure_type": "CUDA_OOM",
    "error_summary": "GPU ran out of memory during batch loading after BATCH_SIZE was increased from 8 to 64.",
    "stack_frames": [
        "src/training/trainer.py:312 in _load_batch",
        "src/training/trainer.py:289 in train_epoch",
        "tests/test_model_training.py:88 in test_full_training_run",
    ],
    "affected_modules": ["src/training/trainer.py"],
    "severity": "CRITICAL",
    "reproducible": True,
    "keywords": ["OOM", "CUDA", "RuntimeError", "batch", "14 GiB"],
}

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

_MOCK_JUDGE = {
    "scores": {
        "accuracy": 9,
        "completeness": 8,
        "actionability": 9,
        "precision": 10,
        "clarity": 9,
    },
    "total": 9.0,
    "pass": True,
    "critique": (
        "The RCA correctly identifies the batch size change as root cause with high confidence. "
        "The recommended fix is specific and immediately actionable. "
        "Preventive measures are concrete and enforceable in CI."
    ),
}


def _mock_response(system: str) -> dict[str, Any]:
    """Pick the right mock payload based on which agent is calling.

    Order matters: judge check must come before rca_synthesizer because
    the judge prompt also contains the word 'RCA'.
    """
    s = system.lower()
    if "impartial evaluator" in s:
        return _MOCK_JUDGE
    if "log analyst" in s or "failure signals" in s:
        return _MOCK_LOG_ANALYSIS
    if "change-impact" in s or "git diff" in s:
        return _MOCK_DIFF_ANALYSIS
    if "root cause analysis" in s or "rca synthesizer" in s or "principal sdet" in s:
        return _MOCK_RCA
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
        payload = _mock_response(system)
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
