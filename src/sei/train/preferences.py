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


_DISP_FLIP = {
    "true_positive": "false_positive",
    "false_positive": "true_positive",
    "benign_true_positive": "true_positive",
    "needs_more_data": "true_positive",
}


def reject_wrong_disposition(gold: str | dict[str, Any]) -> str:
    """Flip disposition toward the common collapse (always true_positive / TP↔FP)."""
    obj = _loads(gold)
    cur = obj.get("incident", {}).get("disposition", "true_positive")
    nxt = _DISP_FLIP.get(cur, "true_positive")
    obj.setdefault("incident", {})["disposition"] = nxt
    obj["incident"]["summary"] = f"Mis-triaged as {nxt} without supporting analysis."
    obj["confidence"] = 0.95
    if nxt == "true_positive":
        obj["recommended_actions"] = [
            {
                "priority": 1,
                "type": "contain",
                "action": "Isolate host immediately",
                "d3fend_id": "D3-ISOL",
            }
        ]
    return _dumps(obj)


def reject_always_true_positive(gold: str | dict[str, Any]) -> str:
    """Force true_positive even when gold is BTP/FP/NMD — mirrors held-out failure mode."""
    obj = _loads(gold)
    obj.setdefault("incident", {})["disposition"] = "true_positive"
    obj["incident"]["summary"] = "Assumed true positive without validating benign/noise cues."
    obj["confidence"] = 0.99
    obj["severity"] = {
        "level": "high",
        "score": 80,
        "rationale": "Defaulted to high severity true positive.",
    }
    return _dumps(obj)


def reject_strip_evidence(gold: str | dict[str, Any]) -> str:
    """Empty evidence despite stage-4 requirement for grounded spans."""
    obj = _loads(gold)
    obj["evidence"] = []
    obj["confidence"] = max(float(obj.get("confidence", 0.5)), 0.9)
    return _dumps(obj)


def reject_wrong_evidence_index(gold: str | dict[str, Any]) -> str:
    """Point evidence at a non-existent telemetry_index / wrong field."""
    obj = _loads(gold)
    obj["evidence"] = [
        {
            "telemetry_index": 99,
            "field": "raw.CommandLine",
            "value": "THIS_STRING_IS_NOT_IN_TELEMETRY_xyz",
            "why": "Wrong window index and fabricated quote",
        }
    ]
    return _dumps(obj)


def reject_nmd_as_true_positive(gold: str | dict[str, Any]) -> str:
    """Reject filling needs_more_data gaps by forcing true_positive."""
    obj = _loads(gold)
    # Always produce the bad pattern even if gold isn't NMD — teaches abstention discipline.
    obj.setdefault("incident", {})["disposition"] = "true_positive"
    obj["incident"]["summary"] = "Forced true_positive despite incomplete telemetry."
    obj["confidence"] = 0.95
    obj["severity"] = {
        "level": "high",
        "score": 75,
        "rationale": "Over-called incomplete signal as confirmed attack.",
    }
    return _dumps(obj)


def reject_fractional_severity_score(gold: str | dict[str, Any]) -> str:
    """Reject severity.score in 0..1 float form (schema requires 0..100 int)."""
    obj = _loads(gold)
    obj.setdefault("severity", {})
    obj["severity"]["score"] = 0.3
    obj["severity"]["level"] = obj["severity"].get("level", "medium")
    return _dumps(obj)


def reject_bad_action_alias(gold: str | dict[str, Any]) -> str:
    """Reject non-ontology action types (containment/inform)."""
    obj = _loads(gold)
    obj["recommended_actions"] = [
        {
            "priority": 1,
            "type": "containment",
            "action": "Isolate immediately",
            "d3fend_id": "D3-ISOL",
        },
        {
            "priority": 2,
            "type": "inform",
            "action": "Tell the SOC",
            "d3fend_id": "D3-AL",
        },
    ]
    return _dumps(obj)


def reject_disposition_in_severity(gold: str | dict[str, Any]) -> str:
    """Reject putting a disposition string into severity.level."""
    obj = _loads(gold)
    disp = obj.get("incident", {}).get("disposition", "benign_true_positive")
    obj.setdefault("severity", {})
    obj["severity"]["level"] = disp
    obj["severity"]["score"] = 50
    return _dumps(obj)


RECIPE_BUILDERS = {
    "over_severity": reject_over_severity,
    "bad_technique": reject_bad_technique,
    "ungrounded_evidence": reject_ungrounded_evidence,
    "action_laundry_list": reject_action_laundry_list,
    "wrong_disposition": reject_wrong_disposition,
    "always_true_positive": reject_always_true_positive,
    "strip_evidence": reject_strip_evidence,
    "wrong_evidence_index": reject_wrong_evidence_index,
    "nmd_as_true_positive": reject_nmd_as_true_positive,
    "fractional_severity_score": reject_fractional_severity_score,
    "bad_action_alias": reject_bad_action_alias,
    "disposition_in_severity": reject_disposition_in_severity,
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
