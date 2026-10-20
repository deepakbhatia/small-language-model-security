"""Deterministic rules on SEI JSON — no planner LLM.

Safety: contain / eradicate never auto-run. Only hunt/notify/tune_detection
may be marked automatable when the model sets automatable=true (or default
for notify/tune when unspecified).
"""

from __future__ import annotations

from typing import Any

from sei.agent.types import AuditEvent, CaseRecord, PendingAction, utc_now

# Types that must always wait for a human, even if model sets automatable=true.
NEVER_AUTO = frozenset({"contain", "eradicate", "recover"})
# Types allowed to auto-queue when automatable is true (or defaulted).
MAY_AUTO = frozenset({"hunt", "notify", "tune_detection"})

MAX_ENRICH_ROUNDS = 2


def _action_automatable(raw: dict[str, Any]) -> bool:
    t = str(raw.get("type", ""))
    if t in NEVER_AUTO:
        return False
    if t not in MAY_AUTO:
        return False
    if "automatable" in raw:
        return bool(raw["automatable"])
    # Default: notify/tune can auto; hunt needs explicit true to auto-run.
    return t in {"notify", "tune_detection"}


def apply_policy(case: CaseRecord) -> CaseRecord:
    """Populate actions / state from current SEI + errors."""
    now = utc_now()
    case.updated_at = now

    if case.errors or case.sei is None:
        case.state = "needs_review"
        case.actions = []
        case.auto_ran = []
        case.audit.append(
            AuditEvent(at=now, kind="policy", detail={"state": case.state, "reason": "verify_failed"})
        )
        return case

    disp = (
        (case.sei.get("incident") or {}).get("disposition")
        if isinstance(case.sei, dict)
        else None
    )
    raw_actions = case.sei.get("recommended_actions") or []
    if not isinstance(raw_actions, list):
        raw_actions = []

    actions: list[PendingAction] = []
    auto_ran: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_actions[:5], start=1):
        if not isinstance(raw, dict):
            continue
        automatable = _action_automatable(raw)
        status = "ran" if automatable else "needs_human"
        aid = f"a{i}"
        pa = PendingAction(
            action_id=aid,
            priority=int(raw.get("priority") or i),
            type=str(raw.get("type") or "hunt"),
            action=str(raw.get("action") or ""),
            status=status,  # type: ignore[arg-type]
            d3fend_id=raw.get("d3fend_id"),
            automatable=automatable,
            decided_at=now if automatable else None,
            decided_by="policy" if automatable else None,
            note="auto-simulated (no external tool wired)" if automatable else None,
        )
        actions.append(pa)
        if automatable:
            auto_ran.append(
                {
                    "action_id": aid,
                    "tool": f"policy.{pa.type}",
                    "status": "ok",
                    "detail": pa.action,
                }
            )

    case.actions = actions
    case.auto_ran = auto_ran

    if disp == "needs_more_data":
        if case.enrich_rounds >= MAX_ENRICH_ROUNDS:
            case.state = "needs_human_context"
        else:
            case.state = "enriching"
    elif any(a.status == "needs_human" for a in actions):
        case.state = "awaiting_approval"
    elif disp in {"false_positive", "benign_true_positive"} and not actions:
        case.state = "closed"
    else:
        case.state = "awaiting_approval" if actions else "closed"

    case.audit.append(
        AuditEvent(
            at=now,
            kind="policy",
            detail={
                "state": case.state,
                "disposition": disp,
                "n_auto": len(auto_ran),
                "n_pending": len(case.pending_approvals()),
            },
        )
    )
    return case
