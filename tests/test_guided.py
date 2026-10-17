"""Constrained-decode readiness checks (no GPU / no model weights)."""

from __future__ import annotations

import pytest

from sei.infer.constrained import (
    GuidedDecodeUnavailable,
    SCHEMA_PATH,
    assert_guided_ready,
    guided_available_transformers,
)


def test_schema_file_present():
    assert SCHEMA_PATH.exists(), f"missing {SCHEMA_PATH}"


def test_assert_guided_ready_transformers():
    if not guided_available_transformers():
        with pytest.raises(GuidedDecodeUnavailable):
            assert_guided_ready(backend="transformers")
    else:
        assert_guided_ready(backend="transformers")


def test_assert_guided_ready_vllm_only_needs_schema():
    assert_guided_ready(backend="vllm")
