"""Compose SEI weak-label examples from SigmaHQ rules (DRL-1.1 / attribution)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from sei.compose.attack_meta import default_actions, normalize_technique_id, technique_obj
from sei.normalize.example import build_example

_TAG_TECH = re.compile(r"attack\.t(\d{4}(?:\.\d{3})?)", re.I)


def _techniques_from_tags(tags: list[Any]) -> list[str]:
    out: list[str] = []
    for t in tags or []:
        s = str(t)
        m = _TAG_TECH.search(s)
        if m:
            tid = normalize_technique_id("T" + m.group(1))
            if tid and tid not in out:
                out.append(tid)
        else:
            tid = normalize_technique_id(s)
            if tid and tid not in out:
                out.append(tid)
    return out[:5]


def _flatten_detection_strings(obj: Any, out: list[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(obj, str):
        if 2 < len(obj) < 200 and not obj.startswith("@"):
            out.append(obj)
    elif isinstance(obj, list):
        for x in obj[:12]:
            _flatten_detection_strings(x, out, depth + 1)
    elif isinstance(obj, dict):
        for k, v in list(obj.items())[:20]:
            if str(k).endswith("|re") or str(k).endswith("|contains"):
                _flatten_detection_strings(v, out, depth + 1)
            else:
                _flatten_detection_strings(v, out, depth + 1)


def _source_from_logsource(logsource: dict[str, Any]) -> str:
    product = str((logsource or {}).get("product") or "").lower()
    category = str((logsource or {}).get("category") or "").lower()
    service = str((logsource or {}).get("service") or "").lower()
    if product in {"windows", "sysmon"} or category in {"process_creation", "ps_script"}:
        return "sysmon"
    if product in {"linux"} or service in {"sshd", "auth"}:
        return "auth"
    if product in {"zeek", "suricata"} or category in {"network", "firewall", "proxy"}:
        return "suricata"
    return "sigma"


def sigma_rule_to_example(doc: dict[str, Any], *, stage: int, path_hint: str = "") -> dict[str, Any] | None:
    tags = doc.get("tags") or []
    tech_ids = _techniques_from_tags(tags if isinstance(tags, list) else [])
    if not tech_ids:
        return None
    title = str(doc.get("title") or "Sigma rule").strip()
    if not title:
        return None

    tokens: list[str] = []
    _flatten_detection_strings(doc.get("detection") or {}, tokens)
    # unique preserve order
    seen: set[str] = set()
    uniq = []
    for t in tokens:
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        uniq.append(t)
    snippet = " | ".join(uniq[:6]) or title
    message = f"{title}: {snippet}"[:500]
    evidence_val = (uniq[0] if uniq else title)[:180]

    techniques = [technique_obj(t, confidence=0.75) for t in tech_ids]
    primary = tech_ids[0]
    level_map = {"informational": ("informational", 15), "low": ("low", 30), "medium": ("medium", 50), "high": ("high", 75), "critical": ("critical", 90)}
    sev = level_map.get(str(doc.get("level") or "medium").lower(), ("medium", 50))
    source = _source_from_logsource(doc.get("logsource") or {})

    output = {
        "incident": {
            "disposition": "true_positive",
            "category": (techniques[0].get("tactic") or "unknown").lower().replace(" ", "_"),
            "summary": f"Sigma detection matched: {title}",
        },
        "techniques": techniques,
        "severity": {
            "level": sev[0],
            "score": sev[1],
            "rationale": f"Mapped from Sigma level={doc.get('level')} for {primary}",
        },
        "confidence": 0.75,
        "evidence": [
            {
                "telemetry_index": 0,
                "field": "message",
                "value": evidence_val,
                "why": "Keyword/field from Sigma detection selection",
            }
        ],
        "recommended_actions": default_actions(primary)[:5]
        or [
            {
                "priority": 1,
                "type": "hunt",
                "action": f"Validate Sigma hit for {primary} and collect corroborating telemetry",
                "d3fend_id": "D3-NTA",
            }
        ],
    }
    event = {
        "timestamp": "2024-07-02T12:00:00Z",
        "source": source,
        "host": "sensor-sigma",
        "raw": {
            "sigma_title": title,
            "sigma_id": doc.get("id"),
            "CommandLine": snippet if source == "sysmon" else None,
            "message": message,
        },
        "message": message,
    }
    event["raw"] = {k: v for k, v in event["raw"].items() if v is not None}

    slug = hashlib.sha1(f"{title}:{primary}".encode()).hexdigest()[:10]
    return build_example(
        example_id=f"sigma-{primary}-{slug}-s{stage}",
        events=[event],
        output=output,
        context={"asset_criticality": "medium", "environment": "prod"},
        meta={
            "source": "sigmahq",
            "license": "DRL-1.1",
            "synthetic": True,
            "split_group": f"sigma-{primary}",
            "sigma_title": title,
            "path": path_hint,
        },
        stage=stage,
    )


def load_sigma_dir(
    rules_dir: Path,
    *,
    stages: list[int],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    root = Path(rules_dir)
    if not root.exists():
        return []
    files = sorted(root.rglob("*.yml")) + sorted(root.rglob("*.yaml"))
    examples: list[dict[str, Any]] = []
    for path in files:
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(doc, dict) or "title" not in doc:
            continue
        for st in stages:
            ex = sigma_rule_to_example(doc, stage=st, path_hint=str(path.relative_to(root)))
            if ex is None:
                continue
            examples.append(ex)
            if limit is not None and len(examples) >= limit:
                return examples
    return examples
