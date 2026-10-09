"""Compose SEI examples from Atomic Red Team YAML tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from sei.compose.attack_meta import default_actions, normalize_technique_id, substitute_atomic_args, technique_obj
from sei.normalize.example import build_example

_EXEC_SOURCE = {
    "command_prompt": ("sysmon", "cmd.exe"),
    "powershell": ("sysmon", "powershell.exe"),
    "sh": ("other", "/bin/sh"),
    "bash": ("other", "/bin/bash"),
}


def _short_cmd(cmd: str, limit: int = 500) -> str:
    cmd = " ".join(cmd.split())
    return cmd if len(cmd) <= limit else cmd[: limit - 3] + "..."


def atomic_test_to_example(
    *,
    technique_id: str,
    display_name: str,
    test: dict[str, Any],
    stage: int,
    idx: int,
) -> dict[str, Any] | None:
    executor = test.get("executor") or {}
    name = str(executor.get("name") or "")
    if name == "manual" or not executor.get("command"):
        return None
    cmd = substitute_atomic_args(str(executor["command"]), test.get("input_arguments"))
    cmd = _short_cmd(cmd)
    if len(cmd) < 3:
        return None

    tid = normalize_technique_id(technique_id)
    if not tid:
        return None

    source, image = _EXEC_SOURCE.get(name, ("other", "unknown"))
    platforms = test.get("supported_platforms") or []
    host = "wkstn-atomic" if "windows" in platforms else "host-atomic"
    test_name = str(test.get("name") or f"test-{idx}")
    evidence_val = cmd[:180]
    tech = technique_obj(tid, confidence=0.85)
    output = {
        "incident": {
            "disposition": "true_positive",
            "category": (tech.get("tactic") or "execution").lower().replace(" ", "_"),
            "summary": f"Atomic Red Team: {test_name} ({tid})",
        },
        "techniques": [tech],
        "severity": {
            "level": "high",
            "score": 75,
            "rationale": f"Atomic simulation of {tid} — {display_name}",
        },
        "confidence": 0.85,
        "evidence": [
            {
                "telemetry_index": 0,
                "field": "raw.CommandLine",
                "value": evidence_val,
                "why": f"Executor command from Atomic test '{test_name}'",
            }
        ],
        "recommended_actions": default_actions(tid)[:5]
        or [
            {
                "priority": 1,
                "type": "hunt",
                "action": f"Hunt for related {tid} activity across estate",
                "d3fend_id": "D3-NTA",
            }
        ],
    }
    event = {
        "timestamp": "2024-07-01T12:00:00Z",
        "source": source,
        "host": host,
        "user": "attacker",
        "raw": {
            "EventID": 1 if source == "sysmon" else None,
            "Image": image,
            "CommandLine": cmd,
            "atomic_test": test_name,
            "attack_technique": tid,
        },
        "message": cmd,
    }
    # drop nulls in raw
    event["raw"] = {k: v for k, v in event["raw"].items() if v is not None}

    slug = hashlib.sha1(f"{tid}:{test_name}:{cmd}".encode()).hexdigest()[:10]
    return build_example(
        example_id=f"atomic-{tid}-{slug}-s{stage}",
        events=[event],
        output=output,
        context={"asset_criticality": "high", "environment": "lab"},
        meta={
            "source": "atomic-red-team",
            "license": "Apache-2.0",
            "synthetic": False,
            "split_group": f"atomic-{tid}",
            "atomic_test": test_name,
        },
        stage=stage,
    )


def load_atomic_dir(
    atomics_dir: Path,
    *,
    stages: list[int],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load all atomics/**/*.yaml under an Atomic Red Team atomics/ folder."""
    root = Path(atomics_dir)
    if not root.exists():
        return []
    files = sorted(root.glob("T*/T*.yaml")) + sorted(root.glob("t*/t*.yaml"))
    # de-dupe case variants
    seen_paths: set[Path] = set()
    examples: list[dict[str, Any]] = []
    for path in files:
        rp = path.resolve()
        if rp in seen_paths:
            continue
        seen_paths.add(rp)
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        tid = doc.get("attack_technique") or path.parent.name.upper()
        display = str(doc.get("display_name") or tid)
        for i, test in enumerate(doc.get("atomic_tests") or []):
            if not isinstance(test, dict):
                continue
            for st in stages:
                ex = atomic_test_to_example(
                    technique_id=str(tid),
                    display_name=display,
                    test=test,
                    stage=st,
                    idx=i,
                )
                if ex is None:
                    continue
                examples.append(ex)
                if limit is not None and len(examples) >= limit:
                    return examples
    return examples
