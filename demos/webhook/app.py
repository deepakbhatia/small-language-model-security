"""FastAPI SIEM webhook demo — accepts alert JSON, returns SEI (stub or model)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from sei.infer.engine import SEIInferencer

app = FastAPI(title="SEI SLM Webhook Demo", version="0.1.0")

_engine: SEIInferencer | None = None


def get_engine() -> SEIInferencer:
    global _engine
    if _engine is None:
        from sei.infer.engine import DEFAULT_ADAPTER, DEFAULT_BASE

        model = os.environ.get("SEI_MODEL_PATH", DEFAULT_BASE)
        adapter = os.environ.get("SEI_ADAPTER_PATH", str(DEFAULT_ADAPTER))
        # SEI_STUB=1 forces stub. Default: use adapter if present, else stub.
        stub_env = os.environ.get("SEI_STUB")
        if stub_env is None:
            stub = not Path(adapter).exists() if adapter else True
        else:
            stub = stub_env == "1"
        if os.environ.get("SEI_NO_ADAPTER", "0") == "1":
            adapter = None
        backend = os.environ.get("SEI_BACKEND", "transformers")
        load_4bit = os.environ.get("SEI_LOAD_4BIT")
        load_in_4bit = None if load_4bit is None else load_4bit == "1"
        _engine = SEIInferencer(
            model_path=model,
            adapter_path=None if stub else adapter,
            backend=backend,
            stub=stub,
            load_in_4bit=load_in_4bit,
        )
    return _engine


class AlertIn(BaseModel):
    events: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    # Convenience single-event fields
    message: str | None = None
    source: str | None = None
    raw: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/triage")
def triage(alert: AlertIn) -> dict[str, Any]:
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
    result = get_engine().generate(events, alert.context or None)
    return {
        "sei": result.get("sei"),
        "errors": result.get("errors", []),
        "stub": result.get("stub", False),
    }


@app.post("/v1/wazuh")
def wazuh_hook(payload: dict[str, Any]) -> dict[str, Any]:
    """Map a minimal Wazuh alert-ish payload into SEI events."""
    data = payload.get("data") or payload
    rule = data.get("rule") or {}
    event = {
        "timestamp": data.get("timestamp") or data.get("@timestamp") or "1970-01-01T00:00:00Z",
        "source": "other",
        "host": (data.get("agent") or {}).get("name"),
        "raw": {
            "rule_id": rule.get("id"),
            "rule_description": rule.get("description"),
            "full_log": data.get("full_log"),
        },
        "message": data.get("full_log") or rule.get("description") or "",
    }
    result = get_engine().generate([event], {"environment": "prod"})
    return {"sei": result.get("sei"), "errors": result.get("errors", []), "stub": result.get("stub", False)}
