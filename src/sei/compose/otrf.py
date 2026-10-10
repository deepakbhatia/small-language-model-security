"""Compose SEI examples from curated OTRF-style scenarios (+ optional JSONL)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from sei.blind_guard import HELD_OUT_META_FLAG
from sei.compose.attack_meta import default_actions, technique_obj
from sei.normalize.example import build_example

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENARIOS = REPO_ROOT / "data" / "otrf" / "scenarios.yaml"


def _scenario_to_example(
    sc: dict[str, Any],
    *,
    stage: int,
    sealed_held_out: bool = False,
) -> dict[str, Any]:
    tech_in = sc["technique"]
    tid = tech_in["id"]
    tech = technique_obj(tid, confidence=float(tech_in.get("confidence", 0.8)))
    tech["name"] = tech_in.get("name") or tech["name"]
    tech["tactic"] = tech_in.get("tactic") or tech["tactic"]
    level, score = sc["severity"]
    output = {
        "incident": {
            "disposition": sc["disposition"],
            "category": sc["category"],
            "summary": sc["summary"],
        },
        "techniques": [tech],
        "severity": {
            "level": level,
            "score": score,
            "rationale": f"OTRF-style scenario severity for {tid}",
        },
        "confidence": float(tech_in.get("confidence", 0.8)),
        "evidence": [
            {
                "telemetry_index": 0,
                "field": sc["evidence_field"],
                "value": sc["evidence_value"],
                "why": sc["evidence_why"],
            }
        ],
        "recommended_actions": default_actions(tid)[:5]
        or [
            {
                "priority": 1,
                "type": "hunt",
                "action": f"Investigate {tid} activity on host",
                "d3fend_id": "D3-NTA",
            }
        ],
    }
    if sc["disposition"] == "needs_more_data":
        output["recommended_actions"] = [
            {
                "priority": 1,
                "type": "hunt",
                "action": "Collect additional process and network telemetry",
                "d3fend_id": "D3-NTA",
            }
        ]
    if sc["disposition"] == "benign_true_positive":
        output["recommended_actions"] = [
            {
                "priority": 1,
                "type": "tune_detection",
                "action": "Tune detection after confirming expected admin activity",
                "d3fend_id": "D3-SCF",
            }
        ]
    if sc["disposition"] == "false_positive":
        output["recommended_actions"] = [
            {
                "priority": 1,
                "type": "tune_detection",
                "action": "Suppress or retune noisy detection after analyst confirm",
                "d3fend_id": "D3-SCF",
            }
        ]
    meta = {
        "source": "blind-held-out" if sealed_held_out else "otrf-curated",
        "license": "Apache-2.0",
        "synthetic": True,
        "split_group": sc["id"],
    }
    if sealed_held_out:
        meta[HELD_OUT_META_FLAG] = True
        meta["split"] = "held_out_never_train"
    return build_example(
        example_id=f"{sc['id']}-s{stage}",
        events=[sc["event"]],
        output=output,
        context={"asset_criticality": "high", "environment": "prod"},
        meta=meta,
        stage=stage,
    )


def load_otrf_scenarios(
    path: Path | None = None,
    *,
    stages: list[int],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    path = path or DEFAULT_SCENARIOS
    if not path.exists():
        return []
    sealed = "blind/held_out" in str(path).replace("\\", "/")
    docs = yaml.safe_load(path.read_text()) or []
    examples: list[dict[str, Any]] = []
    for sc in docs:
        for st in stages:
            examples.append(_scenario_to_example(sc, stage=st, sealed_held_out=sealed))
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


def load_otrf_jsonl(
    path: Path,
    *,
    stages: list[int],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Optional pre-normalized SEI chat JSONL (one example object per line)."""
    if not path.exists():
        return []
    examples: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        # If already a full SEI example, replicate across stages via rebuild when possible
        if "messages" in obj and "meta" in obj:
            for st in stages:
                row = json.loads(json.dumps(obj))
                row["meta"]["curriculum_stage"] = st
                row["id"] = f"{obj.get('id', 'otrf')}-s{st}"
                examples.append(row)
                if limit is not None and len(examples) >= limit:
                    return examples
        elif "technique" in obj and "event" in obj:
            for st in stages:
                examples.append(_scenario_to_example(obj, stage=st))
                if limit is not None and len(examples) >= limit:
                    return examples
    return examples
