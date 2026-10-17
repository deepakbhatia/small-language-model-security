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

INSTALL_HINT = (
    'Install full constrained decode: pip install -e ".[train]" '
    "(or pip install 'lm-format-enforcer>=0.10') and ensure schemas/sei_v1.json exists."
)


class GuidedDecodeUnavailable(RuntimeError):
    """Raised when schema logits are required but lm-format-enforcer / schema is missing."""


def load_sei_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def guided_available_transformers() -> bool:
    if not SCHEMA_PATH.exists():
        return False
    try:
        import lmformatenforcer  # noqa: F401
    except ImportError:
        return False
    return True


def assert_guided_ready(*, backend: str = "transformers") -> None:
    """Fail fast when ship/eval needs schema-constrained decode."""
    if not SCHEMA_PATH.exists():
        raise GuidedDecodeUnavailable(f"missing schema at {SCHEMA_PATH}. {INSTALL_HINT}")
    if backend == "vllm":
        return
    if not guided_available_transformers():
        raise GuidedDecodeUnavailable(
            "lm-format-enforcer not importable — repair-only fallback is not ship-ready. "
            + INSTALL_HINT
        )


def build_transformers_prefix_fn(
    tokenizer: Any,
    *,
    require: bool = False,
) -> Callable | None:
    """Return prefix_allowed_tokens_fn for model.generate, or None if unavailable.

    When require=True, raise GuidedDecodeUnavailable instead of soft-falling back to repair-only.
    """
    if require:
        assert_guided_ready(backend="transformers")
    elif not SCHEMA_PATH.exists():
        return None
    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
    except ImportError:
        if require:
            raise GuidedDecodeUnavailable(INSTALL_HINT) from None
        return None
    schema = load_sei_schema()
    parser = JsonSchemaParser(schema)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
