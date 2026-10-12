"""Best-effort normalization of model SEI JSON before verify.

Fixes common schema slips (action aliases, fractional severity scores, non-string
evidence values) so constrained-decode gaps and soft model errors don't dominate eval.
"""

from __future__ import annotations

import copy
from typing import Any

from sei.verify.validator import ACTION_TYPES, DISPOSITIONS, SEVERITY_LEVELS

ACTION_ALIASES = {
    "containment": "contain",
    "isolate": "contain",
    "isolation": "contain",
    "eradication": "eradicate",
    "recovery": "recover",
    "hunting": "hunt",
    "investigate": "hunt",
    "investigation": "hunt",
    "inform": "notify",
    "alert": "notify",
    "notification": "notify",
    "tune": "tune_detection",
    "tuning": "tune_detection",
    "suppress": "tune_detection",
    "false_positive_tune": "tune_detection",
}

SEVERITY_ALIASES = {
    "info": "informational",
    "informational": "informational",
    "med": "medium",
    "moderate": "medium",
    "crit": "critical",
    "sev_low": "low",
    "sev_high": "high",
}


def repair_sei_obj(obj: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return obj
    out = copy.deepcopy(obj)

    incident = out.get("incident")
    if isinstance(incident, dict):
        disp = incident.get("disposition")
        if isinstance(disp, str):
            disp_l = disp.strip().lower().replace(" ", "_").replace("-", "_")
            if disp_l in DISPOSITIONS:
                incident["disposition"] = disp_l
            elif disp_l in {"tp", "truepositive"}:
                incident["disposition"] = "true_positive"
            elif disp_l in {"fp", "falsepositive"}:
                incident["disposition"] = "false_positive"
            elif disp_l in {"btp", "benign", "benignpositive"}:
                incident["disposition"] = "benign_true_positive"
            elif disp_l in {"nmd", "unknown", "insufficient"}:
                incident["disposition"] = "needs_more_data"

    sev = out.get("severity")
    if isinstance(sev, dict):
        level = sev.get("level")
        if isinstance(level, str):
            key = level.strip().lower().replace(" ", "_")
            sev["level"] = SEVERITY_ALIASES.get(key, key if key in SEVERITY_LEVELS else level)
            # Model sometimes puts a disposition in severity.level
            if sev["level"] in DISPOSITIONS:
                sev["level"] = "medium"
        score = sev.get("score")
        if isinstance(score, float) and 0.0 <= score <= 1.0:
            sev["score"] = int(round(score * 100))
        elif isinstance(score, (int, float)):
            sev["score"] = int(max(0, min(100, round(score))))
        elif isinstance(score, str):
            try:
                f = float(score)
                sev["score"] = int(round(f * 100)) if 0.0 <= f <= 1.0 else int(max(0, min(100, round(f))))
            except ValueError:
                sev["score"] = 50

    conf = out.get("confidence")
    if isinstance(conf, (int, float)):
        out["confidence"] = float(max(0.0, min(1.0, conf)))

    evidence = out.get("evidence")
    if isinstance(evidence, list):
        fixed = []
        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            item = dict(ev)
            if "value" in item and item["value"] is not None and not isinstance(item["value"], str):
                item["value"] = str(item["value"])
            if "telemetry_index" in item and not isinstance(item["telemetry_index"], int):
                try:
                    item["telemetry_index"] = int(item["telemetry_index"])
                except (TypeError, ValueError):
                    item["telemetry_index"] = 0
            fixed.append(item)
        out["evidence"] = fixed

    actions = out.get("recommended_actions")
    if isinstance(actions, list):
        fixed_a = []
        for act in actions[:5]:
            if not isinstance(act, dict):
                continue
            item = dict(act)
            t = item.get("type")
            if isinstance(t, str):
                key = t.strip().lower().replace(" ", "_").replace("-", "_")
                item["type"] = ACTION_ALIASES.get(key, key if key in ACTION_TYPES else t)
            pr = item.get("priority")
            if isinstance(pr, float):
                item["priority"] = int(max(1, min(5, round(pr))))
            elif isinstance(pr, str):
                try:
                    item["priority"] = int(max(1, min(5, int(pr))))
                except ValueError:
                    item["priority"] = 1
            fixed_a.append(item)
        out["recommended_actions"] = fixed_a

    return out
