"""Rules+SEI agent policy and case store (no model weights)."""

from __future__ import annotations

from pathlib import Path

from sei.agent.policy import apply_policy
from sei.agent.service import CaseAgent
from sei.agent.store import CaseStore
from sei.agent.types import CaseRecord, utc_now


def _case_with_sei(sei: dict, *, errors: list[str] | None = None) -> CaseRecord:
    now = utc_now()
    return CaseRecord(
        case_id="sei-test-1",
        state="open",
        created_at=now,
        updated_at=now,
        events=[{"message": "cmd.exe /c whoami"}],
        context={},
        sei=sei,
        errors=errors or [],
        stub=True,
    )


def test_contain_never_auto_even_if_marked_automatable():
    sei = {
        "incident": {"disposition": "true_positive", "category": "x", "summary": "s"},
        "techniques": [],
        "severity": {"level": "high", "score": 80, "rationale": "r"},
        "confidence": 0.9,
        "evidence": [],
        "recommended_actions": [
            {
                "priority": 1,
                "type": "contain",
                "action": "Isolate host",
                "automatable": True,
            },
            {
                "priority": 2,
                "type": "notify",
                "action": "Page on-call",
            },
        ],
    }
    case = apply_policy(_case_with_sei(sei))
    by_type = {a.type: a for a in case.actions}
    assert by_type["contain"].status == "needs_human"
    assert by_type["contain"].automatable is False
    assert by_type["notify"].status == "ran"
    assert case.state == "awaiting_approval"
    assert len(case.pending_approvals()) == 1


def test_nmd_goes_enriching_then_human_context(tmp_path: Path):
    sei = {
        "incident": {"disposition": "needs_more_data", "category": "unknown", "summary": "thin"},
        "techniques": [],
        "severity": {"level": "low", "score": 20, "rationale": "r"},
        "confidence": 0.3,
        "evidence": [],
        "recommended_actions": [
            {"priority": 1, "type": "hunt", "action": "Collect parent", "automatable": True}
        ],
    }
    case = apply_policy(_case_with_sei(sei))
    assert case.state == "enriching"

    case.enrich_rounds = 2
    case = apply_policy(case)
    assert case.state == "needs_human_context"


def test_approve_deny_and_store(tmp_path: Path):
    calls = {"n": 0}

    def fake_generate(events, context=None):
        calls["n"] += 1
        return {
            "sei": {
                "incident": {"disposition": "true_positive", "category": "x", "summary": "s"},
                "techniques": [],
                "severity": {"level": "medium", "score": 50, "rationale": "r"},
                "confidence": 0.7,
                "evidence": [],
                "recommended_actions": [
                    {"priority": 1, "type": "contain", "action": "Isolate", "automatable": False},
                    {"priority": 2, "type": "notify", "action": "Page", "automatable": True},
                ],
            },
            "errors": [],
            "stub": True,
        }

    agent = CaseAgent(fake_generate, store=CaseStore(tmp_path))
    case = agent.open_case([{"message": "x"}], source="test")
    assert case.case_id.startswith("sei-")
    assert len(case.pending_approvals()) == 1
    assert any(a["tool"] == "policy.notify" for a in case.auto_ran)

    case = agent.decide(case.case_id, "a1", approve=True, actor="alice")
    assert case.actions[0].status == "approved"
    assert case.state == "closed"
    assert agent.get(case.case_id) is not None
    assert calls["n"] == 1
