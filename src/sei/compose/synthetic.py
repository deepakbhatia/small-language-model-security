"""Synthetic Atomic/Sigma-style chains for seed and CI data."""

from __future__ import annotations

from typing import Any

import yaml
from pathlib import Path

from sei.normalize.example import build_example

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = REPO_ROOT / "data" / "d3fend" / "action_templates.yaml"

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "atomic-t1059-003-whoami",
        "technique": {
            "id": "T1059.003",
            "name": "Windows Command Shell",
            "tactic": "Execution",
            "confidence": 0.88,
        },
        "disposition": "true_positive",
        "category": "execution",
        "severity": ("high", 75),
        "event": {
            "timestamp": "2024-06-01T12:00:00Z",
            "source": "sysmon",
            "host": "wkstn-042",
            "user": "jdoe",
            "raw": {
                "EventID": 1,
                "Image": "C:\\Windows\\System32\\cmd.exe",
                "ParentImage": "C:\\Users\\jdoe\\payload.exe",
                "CommandLine": "cmd.exe /c whoami",
            },
            "message": "cmd.exe /c whoami",
        },
        "evidence_field": "raw.CommandLine",
        "evidence_value": "cmd.exe /c whoami",
        "evidence_why": "Interactive recon via cmd under non-system parent.",
        "summary": "Suspicious cmd spawned by user payload.",
    },
    {
        "id": "atomic-t1059-001-enc",
        "technique": {
            "id": "T1059.001",
            "name": "PowerShell",
            "tactic": "Execution",
            "confidence": 0.9,
        },
        "disposition": "true_positive",
        "category": "execution",
        "severity": ("critical", 90),
        "event": {
            "timestamp": "2024-06-02T08:15:00Z",
            "source": "sysmon",
            "host": "srv-finance-01",
            "user": "svc_backup",
            "raw": {
                "EventID": 1,
                "Image": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
                "CommandLine": "powershell.exe -NoP -Enc SQBFAFgA",
            },
            "message": "powershell.exe -NoP -Enc SQBFAFgA",
        },
        "evidence_field": "raw.CommandLine",
        "evidence_value": "powershell.exe -NoP -Enc SQBFAFgA",
        "evidence_why": "Encoded PowerShell execution on finance server.",
        "summary": "Encoded PowerShell on high-value host.",
    },
    {
        "id": "sigma-t1110-001-ssh",
        "technique": {
            "id": "T1110.001",
            "name": "Password Guessing",
            "tactic": "Credential Access",
            "confidence": 0.8,
        },
        "disposition": "true_positive",
        "category": "credential_access",
        "severity": ("medium", 55),
        "event": {
            "timestamp": "2024-06-03T03:22:00Z",
            "source": "auth",
            "host": "bastion-1",
            "user": "root",
            "raw": {
                "program": "sshd",
                "message": "Failed password for root from 203.0.113.50 port 52221 ssh2",
                "src_ip": "203.0.113.50",
                "count": 42,
            },
            "message": "Failed password for root from 203.0.113.50 port 52221 ssh2",
        },
        "evidence_field": "raw.message",
        "evidence_value": "Failed password for root from 203.0.113.50 port 52221 ssh2",
        "evidence_why": "Repeated SSH failures indicative of password guessing.",
        "summary": "SSH brute force against bastion.",
    },
    {
        "id": "fp-admin-software-deploy",
        "technique": {
            "id": "T1059.001",
            "name": "PowerShell",
            "tactic": "Execution",
            "confidence": 0.35,
        },
        "disposition": "benign_true_positive",
        "category": "execution",
        "severity": ("low", 20),
        "event": {
            "timestamp": "2024-06-04T10:00:00Z",
            "source": "sysmon",
            "host": "img-builder-02",
            "user": "svc_sccm",
            "raw": {
                "EventID": 1,
                "Image": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
                "ParentImage": "C:\\Program Files\\SCCM\\ccmexec.exe",
                "CommandLine": "powershell.exe -File C:\\Windows\\CCM\\Scripts\\Deploy.ps1",
            },
            "message": "powershell.exe -File C:\\Windows\\CCM\\Scripts\\Deploy.ps1",
        },
        "evidence_field": "raw.ParentImage",
        "evidence_value": "C:\\Program Files\\SCCM\\ccmexec.exe",
        "evidence_why": "Known software deployment parent; expected admin activity.",
        "summary": "SCCM-driven PowerShell deployment — benign true positive.",
    },
    {
        "id": "atomic-t1003-001-lsass",
        "technique": {
            "id": "T1003.001",
            "name": "LSASS Memory",
            "tactic": "Credential Access",
            "confidence": 0.92,
        },
        "disposition": "true_positive",
        "category": "credential_access",
        "severity": ("critical", 95),
        "event": {
            "timestamp": "2024-06-05T19:41:00Z",
            "source": "sysmon",
            "host": "dc01",
            "user": "admin",
            "raw": {
                "EventID": 10,
                "TargetImage": "C:\\Windows\\System32\\lsass.exe",
                "SourceImage": "C:\\Users\\admin\\procdump.exe",
                "GrantedAccess": "0x1010",
            },
            "message": "Process access lsass.exe by procdump.exe",
        },
        "evidence_field": "raw.TargetImage",
        "evidence_value": "C:\\Windows\\System32\\lsass.exe",
        "evidence_why": "Credential dumping tool accessing LSASS.",
        "summary": "LSASS access consistent with credential dumping.",
    },
    {
        "id": "sigma-t1071-001-beacon",
        "technique": {
            "id": "T1071.001",
            "name": "Web Protocols",
            "tactic": "Command and Control",
            "confidence": 0.85,
        },
        "disposition": "true_positive",
        "category": "command_and_control",
        "severity": ("high", 80),
        "event": {
            "timestamp": "2024-06-06T11:05:00Z",
            "source": "suricata",
            "host": "wkstn-077",
            "raw": {
                "rule": "ET MALWARE Cobalt Strike Beacon",
                "dest_ip": "198.51.100.23",
                "message": "ET MALWARE Cobalt Strike Beacon: malleable c2",
            },
            "message": "ET MALWARE Cobalt Strike Beacon: malleable c2",
        },
        "evidence_field": "raw.message",
        "evidence_value": "ET MALWARE Cobalt Strike Beacon: malleable c2",
        "evidence_why": "IDS signature for known C2 framework.",
        "summary": "Cobalt Strike HTTPS beacon detected.",
    },
    {
        "id": "atomic-t1547-001-runkey",
        "technique": {
            "id": "T1547.001",
            "name": "Registry Run Keys / Startup Folder",
            "tactic": "Persistence",
            "confidence": 0.84,
        },
        "disposition": "true_positive",
        "category": "persistence",
        "severity": ("high", 70),
        "event": {
            "timestamp": "2024-06-07T14:20:00Z",
            "source": "sysmon",
            "host": "wkstn-019",
            "user": "jdoe",
            "raw": {
                "EventID": 13,
                "TargetObject": "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\Updater",
                "Details": "C:\\Users\\jdoe\\AppData\\Roaming\\updater.exe",
            },
            "message": "Registry value set CurrentVersion\\Run\\Updater",
        },
        "evidence_field": "raw.TargetObject",
        "evidence_value": "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\Updater",
        "evidence_why": "Persistence via Run key to user-writable path.",
        "summary": "Suspicious Run key persistence.",
    },
    {
        "id": "needs-more-data-partial",
        "technique": {
            "id": "T1082",
            "name": "System Information Discovery",
            "tactic": "Discovery",
            "confidence": 0.4,
        },
        "disposition": "needs_more_data",
        "category": "discovery",
        "severity": ("informational", 15),
        "event": {
            "timestamp": "2024-06-08T09:00:00Z",
            "source": "sysmon",
            "host": "wkstn-003",
            "user": "alice",
            "raw": {
                "EventID": 1,
                "Image": "C:\\Windows\\System32\\systeminfo.exe",
                "CommandLine": "systeminfo.exe",
            },
            "message": "systeminfo.exe",
        },
        "evidence_field": "raw.CommandLine",
        "evidence_value": "systeminfo.exe",
        "evidence_why": "Single discovery binary; context incomplete.",
        "summary": "Ambiguous discovery activity; need more telemetry.",
    },
]


def _actions_for(technique_id: str) -> list[dict[str, Any]]:
    data = yaml.safe_load(TEMPLATES.read_text()) if TEMPLATES.exists() else {}
    return list(data.get(technique_id) or data.get("default") or [])


def scenario_to_example(scenario: dict[str, Any], *, stage: int = 4) -> dict[str, Any]:
    level, score = scenario["severity"]
    tid = scenario["technique"]["id"]
    output = {
        "incident": {
            "disposition": scenario["disposition"],
            "category": scenario["category"],
            "summary": scenario["summary"],
        },
        "techniques": [scenario["technique"]],
        "severity": {
            "level": level,
            "score": score,
            "rationale": f"Rule/scenario severity for {tid}",
        },
        "confidence": scenario["technique"]["confidence"],
        "evidence": [
            {
                "telemetry_index": 0,
                "field": scenario["evidence_field"],
                "value": scenario["evidence_value"],
                "why": scenario["evidence_why"],
            }
        ],
        "recommended_actions": _actions_for(tid)[:5],
    }
    if scenario["disposition"] == "false_positive":
        output["recommended_actions"] = [
            {
                "priority": 1,
                "type": "tune_detection",
                "action": "Suppress or tune noisy detection after analyst confirm",
                "d3fend_id": "D3-SCF",
            }
        ]
    if scenario["disposition"] == "needs_more_data":
        output["recommended_actions"] = [
            {
                "priority": 1,
                "type": "hunt",
                "action": "Collect additional process and network telemetry",
                "d3fend_id": "D3-NTA",
            }
        ]

    return build_example(
        example_id=scenario["id"],
        events=[scenario["event"]],
        output=output,
        context={"asset_criticality": "high", "environment": "prod"},
        meta={
            "source": "synthetic-atomic-sigma",
            "license": "Apache-2.0",
            "synthetic": True,
            "split_group": scenario["id"].rsplit("-", 1)[0],
        },
        stage=stage,
    )


def iter_seed_examples(stages: list[int] | None = None) -> list[dict[str, Any]]:
    stages = stages or [1, 2, 3, 4]
    out: list[dict[str, Any]] = []
    for sc in SCENARIOS:
        for st in stages:
            ex = scenario_to_example(sc, stage=st)
            ex["id"] = f"{sc['id']}-s{st}"
            out.append(ex)
    return out
