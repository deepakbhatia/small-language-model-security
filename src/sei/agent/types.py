"""Case / action types for the rules+SEI agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

CaseState = Literal[
    "open",
    "enriching",
    "awaiting_approval",
    "needs_human_context",
    "closed",
    "needs_review",
]

ActionStatus = Literal["queued_auto", "ran", "needs_human", "approved", "denied", "skipped"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class PendingAction:
    action_id: str
    priority: int
    type: str
    action: str
    status: ActionStatus
    d3fend_id: str | None = None
    automatable: bool = False
    decided_at: str | None = None
    decided_by: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PendingAction:
        return cls(
            action_id=str(d["action_id"]),
            priority=int(d.get("priority", 1)),
            type=str(d.get("type", "hunt")),
            action=str(d.get("action", "")),
            status=d.get("status", "needs_human"),  # type: ignore[arg-type]
            d3fend_id=d.get("d3fend_id"),
            automatable=bool(d.get("automatable", False)),
            decided_at=d.get("decided_at"),
            decided_by=d.get("decided_by"),
            note=d.get("note"),
        )


@dataclass
class AuditEvent:
    at: str
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditEvent:
        return cls(at=str(d["at"]), kind=str(d["kind"]), detail=dict(d.get("detail") or {}))


@dataclass
class CaseRecord:
    case_id: str
    state: CaseState
    created_at: str
    updated_at: str
    events: list[dict[str, Any]]
    context: dict[str, Any]
    sei: dict[str, Any] | None
    errors: list[str]
    stub: bool
    actions: list[PendingAction] = field(default_factory=list)
    auto_ran: list[dict[str, Any]] = field(default_factory=list)
    enrich_rounds: int = 0
    source: str = "triage"
    audit: list[AuditEvent] = field(default_factory=list)

    def pending_approvals(self) -> list[PendingAction]:
        return [a for a in self.actions if a.status == "needs_human"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "events": self.events,
            "context": self.context,
            "sei": self.sei,
            "errors": self.errors,
            "stub": self.stub,
            "actions": [a.to_dict() for a in self.actions],
            "pending_approvals": [a.to_dict() for a in self.pending_approvals()],
            "auto_ran": self.auto_ran,
            "enrich_rounds": self.enrich_rounds,
            "source": self.source,
            "audit": [e.to_dict() for e in self.audit],
        }

    def client_envelope(self) -> dict[str, Any]:
        """What SIEM/SOC clients should consume (omit full audit by default)."""
        d = self.to_dict()
        d.pop("audit", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CaseRecord:
        return cls(
            case_id=str(d["case_id"]),
            state=d["state"],  # type: ignore[arg-type]
            created_at=str(d["created_at"]),
            updated_at=str(d["updated_at"]),
            events=list(d.get("events") or []),
            context=dict(d.get("context") or {}),
            sei=d.get("sei"),
            errors=list(d.get("errors") or []),
            stub=bool(d.get("stub", False)),
            actions=[PendingAction.from_dict(a) for a in (d.get("actions") or [])],
            auto_ran=list(d.get("auto_ran") or []),
            enrich_rounds=int(d.get("enrich_rounds", 0)),
            source=str(d.get("source", "triage")),
            audit=[AuditEvent.from_dict(e) for e in (d.get("audit") or [])],
        )
