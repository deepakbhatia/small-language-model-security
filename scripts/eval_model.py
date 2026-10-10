#!/usr/bin/env python3
"""Run the fine-tuned adapter on golden examples / JSONL and score.

  # 8-file golden smoke
  python scripts/eval_model.py --gold data/golden --out eval/reports/golden.json

  # Broader holdout JSONL (compose first if missing)
  python scripts/compose_all.py --out data/processed/attack --attack-mix \\
    --otrf data/otrf/scenarios.yaml --seed-stages 2,3,4 --attack-stages 2,3,4 \\
    --atomic-limit 4000 --no-seed
  python scripts/eval_model.py \\
    --adapter checkpoints/sei-sft-attack/adapter \\
    --gold data/processed/attack/val_final.jsonl \\
    --pred-dir eval/preds/attack-val \\
    --out eval/reports/attack-val.json \\
    --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sei.eval_metrics import evaluate_paired, _load_sei_from_obj
from sei.infer.engine import DEFAULT_ADAPTER, DEFAULT_BASE, SEIInferencer


def _load_gold_rows(path: Path) -> list[tuple[str, dict]]:
    """Return (id, row) pairs from a JSON file, JSONL file, or directory of JSON."""
    if not path.exists():
        raise SystemExit(
            f"gold path not found: {path}\n"
            "Compose a holdout first, e.g.\n"
            "  python scripts/compose_all.py --out data/processed/attack --attack-mix \\\n"
            "    --otrf data/otrf/scenarios.yaml --attack-stages 2,3,4 --atomic-limit 4000 --no-seed"
        )
    rows: list[tuple[str, dict]] = []
    if path.is_file():
        if path.suffix == ".jsonl":
            for i, line in enumerate(path.read_text().splitlines()):
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows.append((str(obj.get("id", f"row-{i}")), obj))
            return rows
        obj = json.loads(path.read_text())
        return [(str(obj.get("id", path.stem)), obj)]

    for gf in sorted(path.glob("*.json")):
        obj = json.loads(gf.read_text())
        rows.append((str(obj.get("id", gf.stem)), obj))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval fine-tuned SEI adapter on golden/JSONL")
    parser.add_argument("--gold", type=Path, default=Path("data/golden"))
    parser.add_argument("--adapter", default=os.environ.get("SEI_ADAPTER_PATH", str(DEFAULT_ADAPTER)))
    parser.add_argument("--base", default=os.environ.get("SEI_MODEL_PATH", DEFAULT_BASE))
    parser.add_argument("--pred-dir", type=Path, default=Path("eval/preds/stage1"))
    parser.add_argument("--out", type=Path, default=Path("eval/reports/stage1.json"))
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse preds already in --pred-dir (resume after crash)",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    gold_rows = _load_gold_rows(args.gold)
    if args.limit:
        gold_rows = gold_rows[: args.limit]
    if not gold_rows:
        raise SystemExit(f"no golden examples under {args.gold}")

    args.pred_dir.mkdir(parents=True, exist_ok=True)
    need_model = True
    if args.skip_existing:
        # Only load weights if at least one example still needs inference
        need_model = False
        for eid, _row in gold_rows:
            safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in eid)[:120]
            if not (args.pred_dir / f"{safe_name}.json").exists():
                need_model = True
                break

    eng = None
    if need_model:
        eng = SEIInferencer(
            model_path=args.base,
            adapter_path=None if args.stub else args.adapter,
            stub=args.stub,
            load_in_4bit=False if args.no_4bit else None,
        )

    golds: list[dict] = []
    preds: list[dict] = []
    rows: list[dict] = []

    def _metric_pred(pred: dict | None) -> dict:
        if pred is None or not isinstance(pred, dict):
            pred = {
                "incident": {"disposition": "needs_more_data", "category": "unknown", "summary": ""},
                "techniques": [],
                "severity": {"level": "informational", "score": 0, "rationale": ""},
                "confidence": 0.0,
                "evidence": [],
                "recommended_actions": [],
            }
        if not isinstance(pred.get("severity"), dict):
            pred = {**pred, "severity": {"level": "informational", "score": 0, "rationale": ""}}
        if not isinstance(pred.get("incident"), dict):
            pred = {
                **pred,
                "incident": {"disposition": "needs_more_data", "category": "unknown", "summary": ""},
            }
        return pred

    for eid, row in gold_rows:
        gold = _load_sei_from_obj(row)
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in eid)[:120]
        pred_path = args.pred_dir / f"{safe_name}.json"

        if args.skip_existing and pred_path.exists():
            out_obj = json.loads(pred_path.read_text())
            pred = out_obj.get("sei")
            print(f"{eid}: skip-existing errors={len(out_obj.get('errors') or [])}")
        else:
            assert eng is not None
            messages = [m for m in row["messages"] if m["role"] in ("system", "user")]
            try:
                result = eng.generate_chat(messages)
                pred = result.get("sei")
                errors = result.get("errors", [])
                raw = result.get("raw")
                stub = result.get("stub", False)
            except Exception as e:  # noqa: BLE001 — keep batch eval alive on bad outputs
                pred = None
                errors = [f"eval_exception: {type(e).__name__}: {e}"]
                raw = None
                stub = False
                print(f"{eid}: EXCEPTION {e}")
            out_obj = {
                "id": eid,
                "sei": pred,
                "errors": errors,
                "raw": raw,
                "stub": stub,
            }
            pred_path.write_text(json.dumps(out_obj, indent=2) + "\n")
            print(f"{eid}: sei={'ok' if pred else 'FAIL'} errors={len(out_obj['errors'])}")

        golds.append(gold)
        preds.append(_metric_pred(out_obj.get("sei")))
        rows.append(out_obj)

    report = evaluate_paired(golds, preds)
    report["n_files"] = len(gold_rows)
    report["gold"] = str(args.gold)
    report["adapter"] = None if args.stub else args.adapter
    report["json_valid_rate"] = sum(1 for r in rows if r["sei"] is not None) / len(rows)
    report["verifier_clean_rate"] = sum(1 for r in rows if r["sei"] is not None and not r["errors"]) / len(rows)

    text = json.dumps(report, indent=2)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    print(f"wrote preds → {args.pred_dir}")
    print(f"wrote report → {args.out}")


if __name__ == "__main__":
    main()
