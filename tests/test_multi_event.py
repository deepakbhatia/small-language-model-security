"""Multi-event scenario windows compose and verify."""

from __future__ import annotations

from sei.compose.otrf import _scenario_to_example
from sei.verify.validator import verify_example


def test_multi_event_evidence_indices():
    sc = {
        "id": "test-me-window",
        "technique": {"id": "T1059.001", "name": "PowerShell", "tactic": "Execution", "confidence": 0.8},
        "disposition": "true_positive",
        "category": "execution",
        "severity": ["high", 80],
        "events": [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "source": "sysmon",
                "host": "h1",
                "raw": {"message": "first event ALPHA"},
                "message": "first event ALPHA",
            },
            {
                "timestamp": "2024-01-01T00:01:00Z",
                "source": "sysmon",
                "host": "h1",
                "raw": {"message": "second event BETA"},
                "message": "second event BETA",
            },
        ],
        "evidence": [
            {
                "telemetry_index": 0,
                "field": "raw.message",
                "value": "first event ALPHA",
                "why": "step1",
            },
            {
                "telemetry_index": 1,
                "field": "raw.message",
                "value": "second event BETA",
                "why": "step2",
            },
        ],
        "summary": "two-step window",
        "product_surface": "endpoint",
    }
    ex = _scenario_to_example(sc, stage=4)
    assert ex["meta"]["window_size"] == 2
    errs = verify_example(ex["messages"][1]["content"], ex["messages"][2]["content"])
    assert errs == [], errs
