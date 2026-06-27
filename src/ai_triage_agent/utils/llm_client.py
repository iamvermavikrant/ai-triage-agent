"""Thin wrapper around the Anthropic SDK with retry and structured output."""

from __future__ import annotations

import json
import os
from typing import Any

import structlog
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)

_client: Anthropic | None = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


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
    resolved_model = model or os.getenv("TRIAGE_MODEL", "claude-sonnet-4-6")
    log.info("llm_call", model=resolved_model, user_len=len(user))
    message = get_client().messages.create(
        model=resolved_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text


def call_llm_json(
    *,
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Call the LLM and parse the response as JSON."""
    raw = call_llm(system=system, user=user, model=model, max_tokens=max_tokens)
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    return json.loads(cleaned)  # type: ignore[no-any-return]
