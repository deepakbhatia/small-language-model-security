"""Normalize heterogeneous telemetry into ECS-like records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


SOURCES = frozenset(
    {
        "sysmon",
        "windows_security",
        "zeek",
        "suricata",
        "auth",
        "cloudtrail",
        "okta",
        "edr",
        "other",
        "guide",
        "sigma",
    }
)


def normalize_event(event: dict[str, Any], *, default_source: str = "other") -> dict[str, Any]:
    source = str(event.get("source") or event.get("log_source") or default_source).lower()
    if source not in SOURCES:
        source = "other"

    ts = event.get("timestamp") or event.get("@timestamp") or event.get("TimeCreated")
    if isinstance(ts, (int, float)):
        ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    elif not ts:
        ts = datetime.now(tz=timezone.utc).isoformat()

    raw = event.get("raw")
    if raw is None:
        raw = {k: v for k, v in event.items() if k not in {"timestamp", "source", "host", "user", "message"}}
    if not isinstance(raw, dict):
        raw = {"value": raw}

    return {
        "timestamp": str(ts),
        "source": source,
        "host": event.get("host") or event.get("hostname") or event.get("Computer"),
        "user": event.get("user") or event.get("User") or event.get("user.name"),
        "raw": raw,
        "message": event.get("message") or event.get("CommandLine") or event.get("msg"),
    }


def render_telemetry_block(events: list[dict[str, Any]], context: dict[str, Any] | None = None) -> str:
    import json

    normalized = [normalize_event(e) for e in events]
    parts = ["TELEMETRY:"]
    for ev in normalized:
        parts.append(json.dumps(ev, separators=(",", ":")))
    parts.append("CONTEXT:")
    parts.append(json.dumps(context or {}, separators=(",", ":")))
    return "\n".join(parts)
