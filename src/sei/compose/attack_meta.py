"""Shared ATT&CK metadata helpers for Atomic/Sigma/OTRF composers."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CARDS = REPO_ROOT / "data" / "attack" / "v16.1" / "technique_cards.jsonl"
TEMPLATES = REPO_ROOT / "data" / "d3fend" / "action_templates.yaml"

_TECH_RE = re.compile(r"T\d{4}(?:\.\d{3})?", re.I)


@lru_cache(maxsize=1)
def load_technique_cards() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not CARDS.exists():
        return out
    for line in CARDS.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out[obj["id"]] = obj
    return out


def normalize_technique_id(raw: str) -> str | None:
    m = _TECH_RE.search(str(raw).upper().replace("ATTACK.", "").replace("ATACK.", ""))
    if not m:
        return None
    return m.group(0).upper().replace("T0", "T0")  # already upper


def technique_obj(tid: str, *, confidence: float = 0.8) -> dict[str, Any]:
    cards = load_technique_cards()
    card = cards.get(tid) or {}
    return {
        "id": tid,
        "name": card.get("name") or tid,
        "tactic": card.get("tactic") or "unknown",
        "confidence": confidence,
    }


def default_actions(tid: str) -> list[dict[str, Any]]:
    import yaml

    data = yaml.safe_load(TEMPLATES.read_text()) if TEMPLATES.exists() else {}
    return list(data.get(tid) or data.get("default") or [])


def substitute_atomic_args(command: str, input_arguments: dict[str, Any] | None) -> str:
    """Replace #{arg} with Atomic input_arguments defaults."""
    text = command
    for key, spec in (input_arguments or {}).items():
        default = ""
        if isinstance(spec, dict):
            default = str(spec.get("default") if spec.get("default") is not None else "")
        else:
            default = str(spec)
        text = text.replace("#{" + key + "}", default)
    # drop unresolved placeholders to keep evidence grounded
    text = re.sub(r"#\{[^}]+\}", "ARG", text)
    return text.strip()
