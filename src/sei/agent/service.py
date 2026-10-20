"""Case agent: SEI infer + policy + approve/deny + enrich re-triage."""

from __future__ import annotations

import uuid
from typing import Any, Callable

from sei.agent.policy import MAX_ENRICH_ROUNDS, apply_policy
from sei.agent.store import CaseStore
from sei.agent.types import AuditEvent, CaseRecord, utc_now


class CaseAgent:
    def __init__(
        self,
        generate_fn: Callable[..., dict[str, Any]],
        store: CaseStore | None = None,
    ):
        """generate_fn(events, context) -> {sei, errors, stub, ...} like SEIInferencer.generate."""
        self.generate_fn = generate_fn
        self.store = store or CaseStore()

    def _new_id(self) -> str:
        return f"sei-{utc_now()[:10]}-{uuid.uuid4().hex[:8]}"

    def open_case(
        self,
        events: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        *,
        source: str = "triage",
    ) -> CaseRecord:
        context = dict(context or {})
        result = self.generate_fn(events, context)
        now = utc_now()
        case = CaseRecord(
            case_id=self._new_id(),
            state="open",
            created_at=now,
            updated_at=now,
            events=list(events),
            context=context,
            sei=result.get("sei"),
            errors=list(result.get("errors") or []),
            stub=bool(result.get("stub", False)),
            source=source,
            audit=[AuditEvent(at=now, kind="opened", detail={"source": source})],
        )
        apply_policy(case)
        self.store.save(case)
        return case

    def get(self, case_id: str) -> CaseRecord | None:
        return self.store.get(case_id)

    def append_events(
        self,
        case_id: str,
        events: list[dict[str, Any]],
        *,
        retriage: bool = True,
    ) -> CaseRecord:
        case = self.store.get(case_id)
        if case is None:
            raise KeyError(case_id)
        if not events:
            raise ValueError("events must be non-empty")
        case.events.extend(events)
        case.audit.append(
            AuditEvent(at=utc_now(), kind="events_appended", detail={"n": len(events)})
        )
        if retriage:
            if case.enrich_rounds >= MAX_ENRICH_ROUNDS:
                case.state = "needs_human_context"
                case.updated_at = utc_now()
                self.store.save(case)
                return case
            case.enrich_rounds += 1
            result = self.generate_fn(case.events, case.context)
            case.sei = result.get("sei")
            case.errors = list(result.get("errors") or [])
            case.stub = bool(result.get("stub", False))
            case.audit.append(
                AuditEvent(
                    at=utc_now(),
                    kind="retriage",
                    detail={"enrich_rounds": case.enrich_rounds},
                )
            )
            apply_policy(case)
        else:
            case.updated_at = utc_now()
        self.store.save(case)
        return case

    def decide(
        self,
        case_id: str,
        action_id: str,
        *,
        approve: bool,
        actor: str = "analyst",
        note: str | None = None,
    ) -> CaseRecord:
        case = self.store.get(case_id)
        if case is None:
            raise KeyError(case_id)
        target = next((a for a in case.actions if a.action_id == action_id), None)
        if target is None:
            raise KeyError(f"action {action_id}")
        if target.status not in {"needs_human", "queued_auto"}:
            raise ValueError(f"action {action_id} not decidable (status={target.status})")
        now = utc_now()
        target.status = "approved" if approve else "denied"
        target.decided_at = now
        target.decided_by = actor
        target.note = note
        case.audit.append(
            AuditEvent(
                at=now,
                kind="approved" if approve else "denied",
                detail={"action_id": action_id, "actor": actor, "type": target.type},
            )
        )
        # Close when nothing left pending
        if not case.pending_approvals():
            if any(a.status == "approved" and a.type in {"contain", "eradicate"} for a in case.actions):
                case.state = "closed"
            elif all(a.status in {"denied", "ran", "skipped", "approved"} for a in case.actions):
                case.state = "closed"
            else:
                case.state = "closed"
        case.updated_at = now
        self.store.save(case)
        return case
