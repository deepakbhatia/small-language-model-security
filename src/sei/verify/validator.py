"""SEI output verifier — technique IDs, grounded evidence, action ontology."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

T_RE = re.compile(r"^T\d{4}(\.\d{3})?$")
ACTION_TYPES = frozenset(
    {"contain", "eradicate", "recover", "hunt", "notify", "tune_detection"}
)
DISPOSITIONS = frozenset(
    {"true_positive", "false_positive", "benign_true_positive", "needs_more_data"}
)
SEVERITY_LEVELS = frozenset(
    {"informational", "low", "medium", "high", "critical"}
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ATTACK_IDS = REPO_ROOT / "data" / "attack" / "v16.1" / "technique_ids.txt"
DEFAULT_D3FEND_IDS = REPO_ROOT / "data" / "d3fend" / "ids.txt"


@lru_cache(maxsize=4)
def load_id_set(path: str) -> frozenset[str]:
    p = Path(path)
    if not p.exists():
        return frozenset()
    return frozenset(line.strip() for line in p.read_text().splitlines() if line.strip())


def verify_example(
    telemetry_text: str,
    assistant_json: str | dict[str, Any],
    *,
    attack_ids_path: str | Path | None = None,
    d3fend_ids_path: str | Path | None = None,
    require_attack_membership: bool = True,
) -> list[str]:
    """Return a list of error strings; empty means pass."""
    errors: list[str] = []
    if isinstance(assistant_json, dict):
        obj = assistant_json
    else:
        try:
            obj = json.loads(assistant_json)
        except json.JSONDecodeError as e:
            return [f"invalid_json: {e}"]

    if not isinstance(obj, dict):
        return ["invalid_json: root must be object"]

    attack_path = str(attack_ids_path or DEFAULT_ATTACK_IDS)
    d3fend_path = str(d3fend_ids_path or DEFAULT_D3FEND_IDS)
    attack_ids = load_id_set(attack_path)
    d3fend_ids = load_id_set(d3fend_path)

    incident = obj.get("incident") or {}
    if incident.get("disposition") not in DISPOSITIONS:
        errors.append(f"bad_disposition: {incident.get('disposition')}")

    severity = obj.get("severity") or {}
    if severity.get("level") not in SEVERITY_LEVELS:
        errors.append(f"bad_severity_level: {severity.get('level')}")
    score = severity.get("score")
    if not isinstance(score, int) or not (0 <= score <= 100):
        errors.append(f"bad_severity_score: {score}")

    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)) or not (0 <= float(conf) <= 1):
        errors.append(f"bad_confidence: {conf}")

    for t in obj.get("techniques") or []:
        tid = t.get("id", "")
        if not T_RE.match(str(tid)):
            errors.append(f"bad_technique_id_format: {tid}")
        elif require_attack_membership and attack_ids and tid not in attack_ids:
            errors.append(f"bad_technique_id: {tid}")

    for ev in obj.get("evidence") or []:
        val = ev.get("value", "")
        if not val:
            continue
        # Accept raw substring or JSON-escaped form (Windows paths in dumped logs)
        escaped = json.dumps(val)[1:-1]
        if val not in telemetry_text and escaped not in telemetry_text:
            errors.append(f"ungrounded_evidence: {val!r}")

    actions = obj.get("recommended_actions") or []
    if len(actions) > 5:
        errors.append("too_many_actions")
    for act in actions:
        if act.get("type") not in ACTION_TYPES:
            errors.append(f"bad_action_type: {act.get('type')}")
        d3 = act.get("d3fend_id")
        if d3 and d3fend_ids and d3 not in d3fend_ids:
            errors.append(f"bad_d3fend_id: {d3}")

    required = [
        "incident",
        "techniques",
        "severity",
        "confidence",
        "evidence",
        "recommended_actions",
    ]
    for key in required:
        if key not in obj:
            errors.append(f"missing_field: {key}")

    return errors
