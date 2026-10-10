"""Train-safe OOD curriculum must never touch sealed held_out."""

from __future__ import annotations

from pathlib import Path

import pytest

from sei.compose.ood import load_ood_train_scenarios


def test_ood_train_loads_and_stamps_source():
    rows = load_ood_train_scenarios(
        Path("data/ood/train_scenarios.yaml"),
        stages=[4],
        limit=8,
        variants_per=2,
    )
    assert rows
    for r in rows:
        assert r["meta"]["source"] == "ood-train"
        assert r["meta"].get("blind_held_out") is not True
        assert r["meta"].get("split") != "held_out_never_train"


def test_ood_refuses_held_out_path():
    with pytest.raises(SystemExit):
        load_ood_train_scenarios(
            Path("data/blind/held_out/scenarios.yaml"),
            stages=[4],
            limit=1,
        )
