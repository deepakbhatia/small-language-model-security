"""Deterministic rejected responses for ORPO preference pairs."""

from __future__ import annotations

import copy
import json
from typing import Any

SEVERITY_ORDER = ["informational", "low", "medium", "high", "critical"]


def _loads(assistant: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(assistant, dict):
        return copy.deepcopy(assistant)
    return json.loads(assistant)


def _dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def reject_over_severity(gold: str | dict[str, Any]) -> str:
    obj = _loads(gold)
    level = obj["severity"]["level"]
    idx = SEVERITY_ORDER.index(level) if level in SEVERITY_ORDER else 2
    new_idx = min(len(SEVERITY_ORDER) - 1, idx + 2)
    obj["severity"]["level"] = SEVERITY_ORDER[new_idx]
    obj["severity"]["score"] = min(100, int(obj["severity"].get("score", 50)) + 40)
    obj["severity"]["rationale"] = "Critical impact assumed without additional evidence."
    obj["evidence"] = []
    return _dumps(obj)


def reject_bad_technique(gold: str | dict[str, Any]) -> str:
    obj = _loads(gold)
    obj["techniques"] = [
        {
            "id": "T9999.999",
            "name": "Invented Technique",
            "tactic": "Impact",
            "confidence": 0.99,
        }
    ]
    return _dumps(obj)


def reject_ungrounded_evidence(gold: str | dict[str, Any]) -> str:
    obj = _loads(gold)
    obj["evidence"] = [
        {
            "telemetry_index": 0,
            "field": "raw.CommandLine",
            "value": "THIS_STRING_IS_NOT_IN_TELEMETRY_xyz",
            "why": "Fabricated quote",
        }
    ]
    return _dumps(obj)


def reject_action_laundry_list(gold: str | dict[str, Any]) -> str:
    obj = _loads(gold)
    obj["recommended_actions"] = [
        {
            "priority": i,
            "type": "hunt",
            "action": f"Investigate everything option {i}",
            "d3fend_id": "D3-NTA",
        }
        for i in range(1, 9)
    ]
    return _dumps(obj)


RECIPE_BUILDERS = {
    "over_severity": reject_over_severity,
    "bad_technique": reject_bad_technique,
    "ungrounded_evidence": reject_ungrounded_evidence,
    "action_laundry_list": reject_action_laundry_list,
}


def build_preference_row(
    messages: list[dict[str, str]],
    gold_assistant: str,
    recipe: str = "over_severity",
) -> dict[str, Any]:
    builder = RECIPE_BUILDERS[recipe]
    return {
        "prompt_messages": messages[:-1],
        "chosen": gold_assistant,
        "rejected": builder(gold_assistant),
        "recipe": recipe,
    }
