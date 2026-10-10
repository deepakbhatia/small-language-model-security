"""Sealed blind held-out must never enter training."""

from __future__ import annotations

from sei.blind_guard import (
    BlindHeldOutError,
    assert_dataset_not_held_out,
    assert_not_held_out_path,
    is_held_out_path,
)


def test_held_out_path_detection():
    assert is_held_out_path("data/blind/held_out/eval.jsonl")
    assert is_held_out_path("/tmp/proj/data/blind/held_out/eval.jsonl")
    assert not is_held_out_path("data/processed/attack/train.jsonl")
    assert not is_held_out_path("data/blind/scenarios.yaml")
    assert not is_held_out_path("data/ood/train_scenarios.yaml")
    assert not is_held_out_path("data/processed/ood/train.jsonl")


def test_assert_blocks_train_path():
    try:
        assert_not_held_out_path("data/blind/held_out/eval.jsonl", role="train_file")
    except BlindHeldOutError:
        return
    raise AssertionError("expected BlindHeldOutError")


def test_assert_blocks_meta_flag():
    try:
        assert_dataset_not_held_out(
            [{"id": "x", "meta": {"blind_held_out": True, "split": "held_out_never_train"}}]
        )
    except BlindHeldOutError:
        return
    raise AssertionError("expected BlindHeldOutError")
