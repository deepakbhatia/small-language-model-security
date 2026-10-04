"""FastAPI SIEM webhook demo — accepts alert JSON, returns SEI (stub or model)."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from sei.infer.engine import SEIInferencer

app = FastAPI(title="SEI SLM Webhook Demo", version="0.1.0")

_engine: SEIInferencer | None = None


def get_engine() -> SEIInferencer:
    global _engine
    if _engine is None:
        model = os.environ.get("SEI_MODEL_PATH")
        stub = os.environ.get("SEI_STUB", "1") == "1" or not model
        backend = os.environ.get("SEI_BACKEND", "transformers")
        _engine = SEIInferencer(model_path=model, backend=backend, stub=stub)
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
