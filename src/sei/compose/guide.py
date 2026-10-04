"""GUIDE (Microsoft) tabular rows → SEI training examples."""

from __future__ import annotations

from typing import Any

from sei.normalize.example import build_example

DISPOSITION_MAP = {
    "TruePositive": "true_positive",
    "true_positive": "true_positive",
    "FalsePositive": "false_positive",
    "false_positive": "false_positive",
    "BenignPositive": "benign_true_positive",
    "benign_true_positive": "benign_true_positive",
}

SEVERITY_MAP = {
    "Informational": ("informational", 10),
    "Low": ("low", 25),
    "Medium": ("medium", 50),
    "High": ("high", 75),
    "Critical": ("critical", 90),
}


def _technique_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    tid = row.get("TechniqueId") or row.get("MitreTechniques") or row.get("techniques")
    if not tid:
        return []
    if isinstance(tid, list):
        ids = tid
    else:
        ids = [x.strip() for x in str(tid).replace(";", ",").split(",") if x.strip()]
    out = []
    for i in ids[:5]:
        # GUIDE may use namespaced ids; keep Txxxx when present
        if "T" in i:
            start = i.find("T")
            cand = i[start : start + 8] if len(i) >= start + 5 else i[start:]
            # normalize T1059.001-like
            parts = cand.replace(" ", "")
            out.append(
                {
                    "id": parts if parts.startswith("T") else i,
                    "name": row.get("Category", "unknown"),
                    "tactic": row.get("Category", "unknown"),
                    "confidence": 0.7,
                }
            )
    return out


def guide_row_to_example(row: dict[str, Any], *, stage: int = 1, idx: int = 0) -> dict[str, Any]:
    grade = str(row.get("IncidentGrade") or row.get("disposition") or "needs_more_data")
    disposition = DISPOSITION_MAP.get(grade, "needs_more_data")
    sev_raw = str(row.get("Severity") or row.get("severity") or "Medium")
    level, score = SEVERITY_MAP.get(sev_raw, ("medium", 50))

    detector = row.get("DetectorId") or row.get("AlertTitle") or "unknown_detector"
    category = str(row.get("Category") or row.get("EntityType") or "unknown")
    org = row.get("OrgId") or "org"

    event = {
        "timestamp": row.get("Timestamp") or row.get("CreatedTime") or "1970-01-01T00:00:00Z",
        "source": "guide",
        "host": row.get("DeviceName") or row.get("Hostname"),
        "user": row.get("AccountUpn") or row.get("AccountName"),
        "raw": {
            "DetectorId": detector,
            "Category": category,
            "EntityType": row.get("EntityType"),
            "EvidenceRole": row.get("EvidenceRole"),
            "OsFamily": row.get("OsFamily"),
            "AlertTitle": row.get("AlertTitle"),
        },
        "message": f"GUIDE alert detector={detector} category={category}",
    }

    evidence_val = str(detector)
    output = {
        "incident": {
            "disposition": disposition,
            "category": category.lower().replace(" ", "_")[:64],
            "summary": f"GUIDE incident graded {disposition} for {category}",
        },
        "techniques": _technique_from_row(row) if stage >= 2 else [],
        "severity": {
            "level": level,
            "score": score,
            "rationale": f"Mapped from GUIDE severity {sev_raw}",
        },
        "confidence": 0.75 if disposition != "needs_more_data" else 0.4,
        "evidence": (
            [
                {
                    "telemetry_index": 0,
                    "field": "raw.DetectorId",
                    "value": evidence_val,
                    "why": "Primary detector identity from GUIDE row",
                }
            ]
            if stage >= 3
            else []
        ),
        "recommended_actions": (
            [
                {
                    "priority": 1,
                    "type": "hunt" if disposition == "needs_more_data" else "notify",
                    "action": "Follow GUIDE remediation label if present else queue for analyst",
                    "d3fend_id": "D3-AL",
                }
            ]
            if stage >= 4
            else []
        ),
    }

    return build_example(
        example_id=f"guide-{org}-{idx}",
        events=[event],
        output=output,
        context={"asset_criticality": "medium", "environment": "unknown"},
        meta={
            "source": "microsoft-guide",
            "license": "CDLA-Permissive-2.0",
            "synthetic": False,
            "split_group": str(org),
        },
        stage=stage,
    )
