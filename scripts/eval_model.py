#!/usr/bin/env python3
"""Run the fine-tuned adapter on golden examples and score with the eval harness.

  # Full golden eval (needs GPU/enough RAM for Foundation-Sec-8B + adapter)
  python scripts/eval_model.py --gold data/golden --out eval/reports/stage1.json

  # Single file smoke
  python scripts/eval_model.py --gold data/golden/atomic-t1059-003-whoami.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sei.eval_metrics import evaluate_paired, _load_sei_from_obj
from sei.infer.engine import DEFAULT_ADAPTER, DEFAULT_BASE, SEIInferencer


def _gold_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval fine-tuned SEI adapter on golden set")
    parser.add_argument("--gold", type=Path, default=Path("data/golden"))
    parser.add_argument("--adapter", default=os.environ.get("SEI_ADAPTER_PATH", str(DEFAULT_ADAPTER)))
    parser.add_argument("--base", default=os.environ.get("SEI_MODEL_PATH", DEFAULT_BASE))
    parser.add_argument("--pred-dir", type=Path, default=Path("eval/preds/stage1"))
    parser.add_argument("--out", type=Path, default=Path("eval/reports/stage1.json"))
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    files = _gold_files(args.gold)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no golden files under {args.gold}")

    eng = SEIInferencer(
        model_path=args.base,
        adapter_path=None if args.stub else args.adapter,
        stub=args.stub,
        load_in_4bit=False if args.no_4bit else None,
    )

    args.pred_dir.mkdir(parents=True, exist_ok=True)
    golds: list[dict] = []
    preds: list[dict] = []
    rows: list[dict] = []

    for gf in files:
        row = json.loads(gf.read_text())
        gold = _load_sei_from_obj(row)
        messages = [m for m in row["messages"] if m["role"] in ("system", "user")]
        result = eng.generate_chat(messages)
        pred = result.get("sei")
        out_obj = {
            "id": row.get("id", gf.stem),
            "sei": pred,
            "errors": result.get("errors", []),
            "raw": result.get("raw"),
            "stub": result.get("stub", False),
        }
        (args.pred_dir / gf.name).write_text(json.dumps(out_obj, indent=2) + "\n")
        print(f"{gf.name}: sei={'ok' if pred else 'FAIL'} errors={len(out_obj['errors'])}")
        if pred is None:
            # Keep alignment: empty-ish structure so metrics still run
            pred = {
                "incident": {"disposition": "needs_more_data", "category": "unknown", "summary": ""},
                "techniques": [],
                "severity": {"level": "informational", "score": 0, "rationale": ""},
                "confidence": 0.0,
                "evidence": [],
                "recommended_actions": [],
            }
        golds.append(gold)
        preds.append(pred)
        rows.append(out_obj)

    report = evaluate_paired(golds, preds)
    report["n_files"] = len(files)
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
