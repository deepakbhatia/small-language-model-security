# SEI case agent API (rules + SEI)

Thin orchestrator around the triage model. **Contain / eradicate never auto-run.**

## Run

```bash
# Prefer merged ship weights when present
export SEI_REQUIRE_GUIDED=1   # default when not stub
# optional: SEI_STUB=1 for CI without weights
uvicorn demos.webhook.app:app --port 8080
```

`GET /health` reports whether merged `checkpoints/sei-8b-instruct` is selected.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/wazuh` | Wazuh-ish alert → **case envelope** |
| POST | `/v1/triage` | Generic `{events, context}` → case |
| GET | `/v1/cases/{id}` | Fetch case |
| POST | `/v1/cases/{id}/events` | Append telemetry; re-triage (NMD path) |
| POST | `/v1/cases/{id}/approve/{action_id}` | Human approve |
| POST | `/v1/cases/{id}/deny/{action_id}` | Human deny |

## Case envelope (client contract)

```json
{
  "case_id": "sei-2026-10-20-ab12cd34",
  "state": "awaiting_approval",
  "sei": { },
  "errors": [],
  "actions": [],
  "pending_approvals": [],
  "auto_ran": [],
  "enrich_rounds": 0,
  "stub": false
}
```

States: `enriching` | `awaiting_approval` | `needs_human_context` | `needs_review` | `closed`.

## Client flow

1. POST alert → store `case_id` on the SIEM ticket  
2. If `state=enriching`, agent/tools should add events (or client POSTs `/events`)  
3. Show `pending_approvals`; call approve/deny for `contain`  
4. Do not treat `auto_ran` as real EDR isolate — notify/hunt are policy-simulated until tools are wired  

Cases persist under `data/cases/` (`SEI_CASE_DIR` override).
