"""Tests for Atomic/Sigma/OTRF composers."""

from __future__ import annotations

from pathlib import Path

import yaml

from sei.compose.atomic import atomic_test_to_example, load_atomic_dir
from sei.compose.otrf import load_otrf_scenarios
from sei.compose.sigma import sigma_rule_to_example
from sei.verify.validator import verify_example


def test_otrf_scenarios_verify():
    rows = load_otrf_scenarios(stages=[4])
    assert len(rows) >= 8
    for row in rows:
        errs = verify_example(row["messages"][1]["content"], row["messages"][2]["content"])
        assert errs == [], (row["id"], errs)


def test_atomic_test_compose(tmp_path: Path):
    tech_dir = tmp_path / "T1059.003"
    tech_dir.mkdir()
    doc = {
        "attack_technique": "T1059.003",
        "display_name": "Windows Command Shell",
        "atomic_tests": [
            {
                "name": "whoami",
                "supported_platforms": ["windows"],
                "executor": {"name": "command_prompt", "command": "whoami\n"},
            }
        ],
    }
    (tech_dir / "T1059.003.yaml").write_text(yaml.safe_dump(doc))
    rows = load_atomic_dir(tmp_path, stages=[2], limit=5)
    assert rows
    errs = verify_example(rows[0]["messages"][1]["content"], rows[0]["messages"][2]["content"])
    assert errs == []
    assert "T1059.003" in rows[0]["messages"][2]["content"]


def test_sigma_rule_compose():
    doc = {
        "title": "Suspicious Whoami",
        "id": "test-sigma-1",
        "level": "high",
        "logsource": {"product": "windows", "category": "process_creation"},
        "detection": {"selection": {"CommandLine|contains": "whoami"}, "condition": "selection"},
        "tags": ["attack.t1059.003", "attack.execution"],
    }
    ex = sigma_rule_to_example(doc, stage=2)
    assert ex is not None
    errs = verify_example(ex["messages"][1]["content"], ex["messages"][2]["content"])
    assert errs == []
    assert "T1059.003" in ex["messages"][2]["content"]


def test_atomic_skips_manual():
    ex = atomic_test_to_example(
        technique_id="T1059.003",
        display_name="x",
        test={"name": "manual", "executor": {"name": "manual", "steps": "do it"}},
        stage=2,
        idx=0,
    )
    assert ex is None
