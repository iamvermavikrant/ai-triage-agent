"""Loads and renders versioned prompts from config/prompts.yaml."""

from __future__ import annotations

import functools
from pathlib import Path
from string import Template
from typing import Any

import yaml


_PROMPTS_PATH = Path(__file__).parents[3] / "config" / "prompts.yaml"


@functools.lru_cache(maxsize=1)
def _load_raw() -> dict[str, Any]:
    with _PROMPTS_PATH.open() as fh:
        return yaml.safe_load(fh)


def get_system_prompt(key: str) -> str:
    """Return the system prompt for *key* (e.g. 'log_analyzer')."""
    data = _load_raw()
    return data["prompts"][key]["system"].strip()


def render_user_prompt(key: str, **kwargs: str) -> str:
    """Return the user prompt for *key* with variables substituted."""
    data = _load_raw()
    template_str = data["prompts"][key]["user_template"]
    return Template(template_str).safe_substitute(**kwargs).strip()


def prompt_version(key: str) -> str:
    data = _load_raw()
    return data["prompts"][key]["version"]
