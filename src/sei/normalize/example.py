"""Build SEI training messages from structured fields."""

from __future__ import annotations

import json
from typing import Any

from sei import SYSTEM_PROMPT
from sei.normalize.ecs import render_telemetry_block


def compact_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def stage_filter_output(output: dict[str, Any], stage: int) -> dict[str, Any]:
    """Reduce assistant payload to curriculum stage fields."""
    if stage <= 1:
        return {
            "incident": {
                "disposition": output["incident"]["disposition"],
                "category": output["incident"].get("category", "unknown"),
                "summary": output["incident"].get("summary", ""),
            },
            "techniques": [],
            "severity": output["severity"],
            "confidence": output.get("confidence", 0.5),
            "evidence": [],
            "recommended_actions": [],
        }
    if stage == 2:
        out = stage_filter_output(output, 1)
        out["techniques"] = output.get("techniques") or []
        return out
    if stage == 3:
        out = stage_filter_output(output, 2)
        out["evidence"] = output.get("evidence") or []
        return out
    return output


def build_example(
    *,
    example_id: str,
    events: list[dict[str, Any]],
    output: dict[str, Any],
    context: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    stage: int = 4,
    rag_cards: str | None = None,
) -> dict[str, Any]:
    user = render_telemetry_block(events, context)
    if rag_cards:
        user = user + "\nATT&CK_CONTEXT:\n" + rag_cards

    staged = stage_filter_output(output, stage)
    meta_out = {
        "attack_version": "16.1",
        "source": "unknown",
        "license": "Apache-2.0",
        "synthetic": True,
        "curriculum_stage": stage,
        "split_group": example_id,
        **(meta or {}),
    }
    meta_out["curriculum_stage"] = stage

    return {
        "id": example_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": compact_json(staged)},
        ],
        "meta": meta_out,
    }
