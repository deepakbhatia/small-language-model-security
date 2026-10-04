"""Evaluation metrics for SEI structured outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import f1_score, precision_recall_fscore_support

SEVERITY_ORDER = ["informational", "low", "medium", "high", "critical"]


def _load_sei(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if "messages" in obj:
        return json.loads(obj["messages"][-1]["content"])
    if "sei" in obj:
        return obj["sei"]
    return obj


def disposition_metrics(golds: list[dict], preds: list[dict]) -> dict[str, Any]:
    y_true = [g["incident"]["disposition"] for g in golds]
    y_pred = [p["incident"]["disposition"] for p in preds]
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = {
        lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
        for i, lab in enumerate(labels)
    }
    return {"macro_f1": float(macro), "per_class": per_class, "support": dict(Counter(y_true))}


def technique_f1(golds: list[dict], preds: list[dict]) -> dict[str, Any]:
    """Micro multi-label exact-ID F1."""
    tp = fp = fn = 0
    tactic_correct = tactic_total = 0
    for g, p in zip(golds, preds):
        gt = {t["id"] for t in g.get("techniques") or []}
        pr = {t["id"] for t in p.get("techniques") or []}
        tp += len(gt & pr)
        fp += len(pr - gt)
        fn += len(gt - pr)
        gt_tac = {t.get("tactic") for t in g.get("techniques") or []}
        pr_tac = {t.get("tactic") for t in p.get("techniques") or []}
        if gt_tac:
            tactic_total += 1
            if gt_tac & pr_tac:
                tactic_correct += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "exact_id_precision": prec,
        "exact_id_recall": rec,
        "exact_id_f1": f1,
        "parent_tactic_accuracy": (tactic_correct / tactic_total) if tactic_total else 0.0,
    }


def severity_metrics(golds: list[dict], preds: list[dict]) -> dict[str, Any]:
    correct = 0
    mae = 0.0
    n = len(golds)
    for g, p in zip(golds, preds):
        gl = g["severity"]["level"]
        pl = p["severity"]["level"]
        if gl == pl:
            correct += 1
        gi = SEVERITY_ORDER.index(gl) if gl in SEVERITY_ORDER else 2
        pi = SEVERITY_ORDER.index(pl) if pl in SEVERITY_ORDER else 2
        mae += abs(gi - pi)
    return {"accuracy": correct / n if n else 0.0, "ordinal_mae": mae / n if n else 0.0}


def evidence_span_f1(golds: list[dict], preds: list[dict]) -> dict[str, float]:
    tp = fp = fn = 0
    for g, p in zip(golds, preds):
        gt = {e.get("value", "") for e in g.get("evidence") or [] if e.get("value")}
        pr = {e.get("value", "") for e in p.get("evidence") or [] if e.get("value")}
        tp += len(gt & pr)
        fp += len(pr - gt)
        fn += len(gt - pr)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def action_validity(preds: list[dict], d3fend_ids: set[str] | None = None) -> dict[str, float]:
    total = 0
    valid_type = 0
    valid_d3 = 0
    overlong = 0
    types = {"contain", "eradicate", "recover", "hunt", "notify", "tune_detection"}
    for p in preds:
        acts = p.get("recommended_actions") or []
        if len(acts) > 5:
            overlong += 1
        for a in acts:
            total += 1
            if a.get("type") in types:
                valid_type += 1
            d3 = a.get("d3fend_id")
            if not d3 or not d3fend_ids or d3 in d3fend_ids:
                valid_d3 += 1
    return {
        "valid_type_rate": valid_type / total if total else 1.0,
        "valid_d3fend_rate": valid_d3 / total if total else 1.0,
        "overlong_action_rate": overlong / len(preds) if preds else 0.0,
    }


def evaluate_paired(golds: list[dict], preds: list[dict], d3fend_ids: set[str] | None = None) -> dict[str, Any]:
    assert len(golds) == len(preds)
    return {
        "n": len(golds),
        "disposition": disposition_metrics(golds, preds),
        "techniques": technique_f1(golds, preds),
        "severity": severity_metrics(golds, preds),
        "evidence": evidence_span_f1(golds, preds),
        "actions": action_validity(preds, d3fend_ids),
    }


def evaluate_dirs(gold_dir: Path, pred_dir: Path) -> dict[str, Any]:
    gold_files = sorted(gold_dir.glob("*.json")) + sorted(gold_dir.glob("*.jsonl"))
    # For directory of json examples
    golds: list[dict] = []
    preds: list[dict] = []
    if gold_dir.is_file():
        # single jsonl
        for line in gold_dir.read_text().splitlines():
            if line.strip():
                golds.append(_load_sei_from_obj(json.loads(line)))
        for line in pred_dir.read_text().splitlines():
            if line.strip():
                preds.append(_load_sei_from_obj(json.loads(line)))
    else:
        files = sorted(p for p in gold_dir.glob("*.json"))
        for gf in files:
            pf = pred_dir / gf.name
            if not pf.exists():
                continue
            golds.append(_load_sei(gf))
            preds.append(_load_sei(pf))
    d3 = set()
    d3_path = Path(__file__).resolve().parents[3] / "data" / "d3fend" / "ids.txt"
    if d3_path.exists():
        d3 = {l.strip() for l in d3_path.read_text().splitlines() if l.strip()}
    return evaluate_paired(golds, preds, d3)


def _load_sei_from_obj(obj: dict) -> dict:
    if "messages" in obj:
        return json.loads(obj["messages"][-1]["content"])
    if "sei" in obj:
        return obj["sei"]
    return obj
