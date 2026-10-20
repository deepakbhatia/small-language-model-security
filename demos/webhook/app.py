"""SEI case agent API — Wazuh/generic triage with rules+SEI policy.

Env:
  SEI_MODEL_PATH   default: checkpoints/sei-8b-instruct if present, else Foundation-Sec-8B
  SEI_ADAPTER_PATH LoRA path (ignored when merged weights used / SEI_NO_ADAPTER=1)
  SEI_NO_ADAPTER   1 = never load LoRA (default 1 when merged dir exists)
  SEI_REQUIRE_GUIDED 1 = fail if schema logits missing (default 1 when not stub)
  SEI_STUB         1 force stub; 0 force weights; unset = stub if no weights found
  SEI_CASE_DIR     case JSON directory (default data/cases)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sei.agent.service import CaseAgent
from sei.infer.engine import SEIInferencer

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DEFAULT = REPO_ROOT / "checkpoints" / "sei-8b-instruct"

app = FastAPI(title="SEI Case Agent", version="0.2.0")

_engine: SEIInferencer | None = None
_agent: CaseAgent | None = None


def _resolve_paths() -> tuple[str, str | None, bool]:
    """Return (model_path, adapter_path|None, prefer_merged)."""
    from sei.infer.engine import DEFAULT_ADAPTER, DEFAULT_BASE

    merged = Path(os.environ.get("SEI_MERGED_PATH", str(MERGED_DEFAULT)))
    if merged.is_dir() and (merged / "config.json").exists():
        return str(merged), None, True

    model = os.environ.get("SEI_MODEL_PATH", DEFAULT_BASE)
    adapter: str | None = os.environ.get("SEI_ADAPTER_PATH", str(DEFAULT_ADAPTER))
    if os.environ.get("SEI_NO_ADAPTER", "0") == "1":
        adapter = None
    return model, adapter, False


def get_engine() -> SEIInferencer:
    global _engine
    if _engine is None:
        model, adapter, prefer_merged = _resolve_paths()
        stub_env = os.environ.get("SEI_STUB")
        if stub_env is None:
            if prefer_merged:
                stub = False
            elif adapter and Path(adapter).exists():
                stub = False
            else:
                stub = True
        else:
            stub = stub_env == "1"

        if stub:
            require_guided = False
        else:
            require_guided = os.environ.get("SEI_REQUIRE_GUIDED", "1") == "1"

        backend = os.environ.get("SEI_BACKEND", "transformers")
        load_4bit = os.environ.get("SEI_LOAD_4BIT")
        load_in_4bit = None if load_4bit is None else load_4bit == "1"
        _engine = SEIInferencer(
            model_path=model,
            adapter_path=None if stub else adapter,
            backend=backend,
            stub=stub,
            load_in_4bit=load_in_4bit,
            require_guided=require_guided,
        )
    return _engine


def get_agent() -> CaseAgent:
    global _agent
    if _agent is None:
        eng = get_engine()

        def _gen(events: list[dict[str, Any]], context: dict[str, Any] | None = None) -> dict[str, Any]:
            return eng.generate(events, context)

        _agent = CaseAgent(_gen)
    return _agent


class AlertIn(BaseModel):
    events: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
    source: str | None = None
    raw: dict[str, Any] | None = None


class EventsIn(BaseModel):
    events: list[dict[str, Any]]
    retriage: bool = True


class DecideIn(BaseModel):
    actor: str = "analyst"
    note: str | None = None


def _events_from_alert(alert: AlertIn) -> list[dict[str, Any]]:
    events = list(alert.events)
    if not events:
        events = [
            {
                "timestamp": "1970-01-01T00:00:00Z",
                "source": alert.source or "other",
                "raw": alert.raw or {},
                "message": alert.message or "",
            }
        ]
    return events


def wazuh_to_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") or payload
    rule = data.get("rule") or {}
    return [
        {
            "timestamp": data.get("timestamp") or data.get("@timestamp") or "1970-01-01T00:00:00Z",
            "source": "wazuh",
            "host": (data.get("agent") or {}).get("name"),
            "raw": {
                "rule_id": rule.get("id"),
                "rule_description": rule.get("description"),
                "full_log": data.get("full_log"),
                **(
                    {"CommandLine": ((data.get("data") or {}).get("win") or {}).get("eventdata", {}).get("CommandLine")}
                    if isinstance(data.get("data"), dict)
                    else {}
                ),
            },
            "message": data.get("full_log") or rule.get("description") or "",
        }
    ]


@app.get("/health")
def health() -> dict[str, Any]:
    model, adapter, prefer_merged = _resolve_paths()
    return {
        "status": "ok",
        "merged": prefer_merged,
        "model": model,
        "adapter": adapter,
    }


@app.post("/v1/triage")
def triage(alert: AlertIn) -> dict[str, Any]:
    """Open a case from generic events (returns case envelope)."""
    case = get_agent().open_case(_events_from_alert(alert), alert.context or None, source="triage")
    return case.client_envelope()


@app.post("/v1/wazuh")
def wazuh_hook(payload: dict[str, Any]) -> dict[str, Any]:
    """Wazuh-ish alert → case envelope."""
    case = get_agent().open_case(wazuh_to_events(payload), {"environment": "prod"}, source="wazuh")
    return case.client_envelope()


@app.get("/v1/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    case = get_agent().get(case_id)
    if case is None:
        raise HTTPException(404, f"case not found: {case_id}")
    return case.client_envelope()


@app.post("/v1/cases/{case_id}/events")
def append_events(case_id: str, body: EventsIn) -> dict[str, Any]:
    try:
        case = get_agent().append_events(case_id, body.events, retriage=body.retriage)
    except KeyError:
        raise HTTPException(404, f"case not found: {case_id}") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return case.client_envelope()


@app.post("/v1/cases/{case_id}/approve/{action_id}")
def approve(case_id: str, action_id: str, body: DecideIn | None = None) -> dict[str, Any]:
    body = body or DecideIn()
    try:
        case = get_agent().decide(
            case_id, action_id, approve=True, actor=body.actor, note=body.note
        )
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return case.client_envelope()


@app.post("/v1/cases/{case_id}/deny/{action_id}")
def deny(case_id: str, action_id: str, body: DecideIn | None = None) -> dict[str, Any]:
    body = body or DecideIn()
    try:
        case = get_agent().decide(
            case_id, action_id, approve=False, actor=body.actor, note=body.note
        )
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return case.client_envelope()
