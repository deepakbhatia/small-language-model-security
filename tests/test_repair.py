"""SEI JSON repair + preference schema rejects."""

from __future__ import annotations

import json

from sei.infer.repair import repair_sei_obj
from sei.train.preferences import (
    reject_bad_action_alias,
    reject_fractional_severity_score,
    reject_nmd_as_true_positive,
)


def test_repair_action_alias_and_fractional_score():
    obj = {
        "incident": {"disposition": "true_positive", "category": "x", "summary": "s"},
        "techniques": [],
        "severity": {"level": "high", "score": 0.3, "rationale": "r"},
        "confidence": 0.9,
        "evidence": [{"telemetry_index": 0, "field": "m", "value": 31, "why": "c"}],
        "recommended_actions": [
            {"priority": 1, "type": "containment", "action": "iso", "d3fend_id": "D3-ISOL"}
        ],
    }
    fixed = repair_sei_obj(obj)
    assert fixed["severity"]["score"] == 30
    assert fixed["evidence"][0]["value"] == "31"
    assert fixed["recommended_actions"][0]["type"] == "contain"


def test_reject_nmd_and_schema_recipes():
    gold = {
        "incident": {"disposition": "needs_more_data", "category": "x", "summary": "s"},
        "techniques": [],
        "severity": {"level": "low", "score": 20, "rationale": "r"},
        "confidence": 0.4,
        "evidence": [],
        "recommended_actions": [],
    }
    bad = json.loads(reject_nmd_as_true_positive(gold))
    assert bad["incident"]["disposition"] == "true_positive"
    frac = json.loads(reject_fractional_severity_score(gold))
    assert frac["severity"]["score"] == 0.3
    alias = json.loads(reject_bad_action_alias(gold))
    assert alias["recommended_actions"][0]["type"] == "containment"
