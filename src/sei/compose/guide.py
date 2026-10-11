"""GUIDE (Microsoft) tabular rows → SEI training examples."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
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

GUIDE_DISPOSITIONS = (
    "true_positive",
    "false_positive",
    "benign_true_positive",
    "needs_more_data",
)

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


def _disposition_of_row(row: dict[str, Any]) -> str:
    grade = str(row.get("IncidentGrade") or row.get("disposition") or "needs_more_data")
    return DISPOSITION_MAP.get(grade, "needs_more_data")


def _event_from_row(row: dict[str, Any]) -> dict[str, Any]:
    detector = row.get("DetectorId") or row.get("AlertTitle") or "unknown_detector"
    category = str(row.get("Category") or row.get("EntityType") or "unknown")
    return {
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


def guide_row_to_example(row: dict[str, Any], *, stage: int = 1, idx: int = 0) -> dict[str, Any]:
    return guide_rows_to_example([row], stage=stage, idx=idx)


def guide_rows_to_example(
    rows: list[dict[str, Any]],
    *,
    stage: int = 1,
    idx: int = 0,
) -> dict[str, Any]:
    """One or more GUIDE alert rows → a (possibly multi-event) SEI example."""
    if not rows:
        raise ValueError("guide_rows_to_example requires at least one row")
    primary = rows[0]
    disposition = _disposition_of_row(primary)
    # If alerts in the window disagree on grade, prefer needs_more_data over forcing TP
    disps = {_disposition_of_row(r) for r in rows}
    if len(disps) > 1:
        disposition = "needs_more_data" if "needs_more_data" in disps else sorted(disps)[0]

    sev_raw = str(primary.get("Severity") or primary.get("severity") or "Medium")
    level, score = SEVERITY_MAP.get(sev_raw, ("medium", 50))
    category = str(primary.get("Category") or primary.get("EntityType") or "unknown")
    org = primary.get("OrgId") or "org"
    incident = primary.get("IncidentId") or primary.get("Id") or idx

    events = [_event_from_row(r) for r in rows]
    evidence = []
    if stage >= 3:
        for ti, ev in enumerate(events):
            evidence.append(
                {
                    "telemetry_index": ti,
                    "field": "raw.DetectorId",
                    "value": str(ev["raw"]["DetectorId"]),
                    "why": "Primary detector identity from GUIDE alert window",
                }
            )

    techs: list[dict[str, Any]] = []
    if stage >= 2:
        seen: set[str] = set()
        for r in rows:
            for t in _technique_from_row(r):
                if t["id"] not in seen:
                    seen.add(t["id"])
                    techs.append(t)

    n_alerts = len(events)
    output = {
        "incident": {
            "disposition": disposition,
            "category": category.lower().replace(" ", "_")[:64],
            "summary": (
                f"GUIDE incident graded {disposition} for {category}"
                + (f" ({n_alerts}-alert window)" if n_alerts > 1 else "")
            ),
        },
        "techniques": techs,
        "severity": {
            "level": level,
            "score": score,
            "rationale": f"Mapped from GUIDE severity {sev_raw}",
        },
        "confidence": 0.75 if disposition != "needs_more_data" else 0.4,
        "evidence": evidence,
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
        example_id=f"guide-{org}-{incident}-{idx}",
        events=events,
        output=output,
        context={"asset_criticality": "medium", "environment": "unknown"},
        meta={
            "source": "microsoft-guide",
            "license": "CDLA-Permissive-2.0",
            "synthetic": False,
            "split_group": str(org),
            "window_size": n_alerts,
        },
        stage=stage,
    )


def load_guide_balanced(
    path: Path,
    *,
    per_class: int,
    stage: int,
    offset: int = 0,
    scan_limit: int | None = None,
    window_size: int = 1,
) -> list[dict[str, Any]]:
    """Stratified GUIDE sample: up to per_class examples per disposition.

    When window_size > 1, groups alerts by IncidentId into multi-event windows
    (still counted toward the primary row's disposition bucket).
    """
    per_class = max(0, int(per_class))
    window_size = max(1, int(window_size))
    if per_class == 0:
        return []

    # First pass: collect raw rows into incident buckets (or singles)
    incident_rows: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    singles: list[tuple[int, dict[str, Any]]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < offset:
                continue
            if scan_limit is not None and (i - offset) >= scan_limit:
                break
            if window_size > 1:
                key = str(row.get("IncidentId") or row.get("Id") or f"row-{i}")
                incident_rows[key].append((i, row))
            else:
                singles.append((i, row))

    buckets: dict[str, list[dict[str, Any]]] = {d: [] for d in GUIDE_DISPOSITIONS}

    def _try_add(ex: dict[str, Any], disp: str) -> None:
        if disp not in buckets:
            disp = "needs_more_data"
        if len(buckets[disp]) < per_class:
            buckets[disp].append(ex)

    if window_size > 1:
        for key, items in incident_rows.items():
            items = sorted(items, key=lambda t: t[0])[:window_size]
            if len(buckets[_disposition_of_row(items[0][1])]) >= per_class:
                # still allow if another class needs fill — check primary
                pass
            rows = [r for _, r in items]
            disp = _disposition_of_row(rows[0])
            if len(buckets.get(disp, [])) >= per_class:
                continue
            ex = guide_rows_to_example(rows, stage=stage, idx=items[0][0])
            _try_add(ex, disp)
            if all(len(buckets[d]) >= per_class for d in GUIDE_DISPOSITIONS):
                break
    else:
        for i, row in singles:
            disp = _disposition_of_row(row)
            if len(buckets.get(disp, buckets["needs_more_data"])) >= per_class:
                if all(len(buckets[d]) >= per_class for d in GUIDE_DISPOSITIONS):
                    break
                continue
            ex = guide_rows_to_example([row], stage=stage, idx=i)
            _try_add(ex, disp)

    out: list[dict[str, Any]] = []
    for d in GUIDE_DISPOSITIONS:
        out.extend(buckets[d])
    return out
