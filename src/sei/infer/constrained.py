"""Constrained / guided decoding helpers for SEI JSON.

vLLM: GuidedDecodingParams(json=schema) via engine.
transformers: lm-format-enforcer JsonSchemaParser prefix_allowed_tokens_fn when installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "sei_v1.json"


def load_sei_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def build_transformers_prefix_fn(tokenizer: Any) -> Callable | None:
    """Return prefix_allowed_tokens_fn for model.generate, or None if unavailable."""
    if not SCHEMA_PATH.exists():
        return None
    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
    except ImportError:
        return None
    schema = load_sei_schema()
    parser = JsonSchemaParser(schema)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)


def guided_available_transformers() -> bool:
    try:
        import lmformatenforcer  # noqa: F401

        return SCHEMA_PATH.exists()
    except ImportError:
        return False
