"""Train-safe OOD curriculum (SaaS/cloud/IdP + BTP/FP/needs_more_data).

NEVER load data/blind/held_out/ here — that tree is sealed eval-only.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml

from sei.blind_guard import is_held_out_path
from sei.compose.otrf import _scenario_to_example

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OOD = REPO_ROOT / "data" / "ood" / "train_scenarios.yaml"

_HOST_VARIANTS = ("wrk-ood-01", "wrk-ood-02", "srv-ood-a", "srv-ood-b", "edge-ood-3")
_USER_VARIANTS = ("alice.ood", "bob.ood", "svc_ood", "contractor_ood", "admin_ood")


def _variant(sc: dict[str, Any], *, host: str, user: str, idx: int) -> dict[str, Any]:
    out = copy.deepcopy(sc)
    out["id"] = f"{sc['id']}-v{idx}"
    ev = out.setdefault("event", {})
    ev["host"] = host
    if "user" in ev or user:
        ev["user"] = user
    # keep evidence_value grounded: do not rewrite evidence strings unless present in message/raw
    return out


def load_ood_train_scenarios(
    path: Path | None = None,
    *,
    stages: list[int],
    limit: int | None = None,
    variants_per: int = 3,
) -> list[dict[str, Any]]:
    path = path or DEFAULT_OOD
    if is_held_out_path(path):
        raise SystemExit(f"Refusing OOD train load of sealed held_out path: {path}")
    if not path.exists():
        return []
    docs = yaml.safe_load(path.read_text()) or []
    examples: list[dict[str, Any]] = []
    for sc in docs:
        base_variants = [sc]
        for i in range(max(0, variants_per - 1)):
            base_variants.append(
                _variant(
                    sc,
                    host=_HOST_VARIANTS[i % len(_HOST_VARIANTS)],
                    user=_USER_VARIANTS[i % len(_USER_VARIANTS)],
                    idx=i + 1,
                )
            )
        for variant in base_variants:
            for st in stages:
                ex = _scenario_to_example(variant, stage=st, sealed_held_out=False)
                # Force train-safe provenance (never held_out flags)
                ex["meta"]["source"] = "ood-train"
                ex["meta"]["license"] = "Apache-2.0"
                ex["meta"]["synthetic"] = True
                ex["meta"].pop("blind_held_out", None)
                if ex["meta"].get("split") == "held_out_never_train":
                    ex["meta"]["split"] = "ood_train"
                # stable id
                digest = hashlib.sha1(ex["id"].encode()).hexdigest()[:8]
                ex["id"] = f"{ex['id']}-{digest}"
                examples.append(ex)
                if limit is not None and len(examples) >= limit:
                    return examples
    return examples
