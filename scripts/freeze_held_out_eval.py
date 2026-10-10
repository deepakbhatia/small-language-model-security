#!/usr/bin/env python3
"""Freeze the sealed blind held-out eval set (NEVER used for training).

  python scripts/freeze_held_out_eval.py
  # → data/blind/held_out/eval.jsonl  (git-tracked)

  python scripts/eval_model.py \\
    --adapter checkpoints/sei-sft-attack/adapter \\
    --gold data/blind/held_out/eval.jsonl \\
    --pred-dir eval/preds/held-out \\
    --out eval/reports/held-out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sei.blind_guard import HELD_OUT_META_FLAG
from sei.compose.otrf import load_otrf_scenarios
from sei.verify.validator import verify_example

ROOT = Path(__file__).resolve().parents[1]
HELD = ROOT / "data" / "blind" / "held_out"
SCENARIOS = HELD / "scenarios.yaml"
OUT = HELD / "eval.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--stage", type=int, default=4)
    args = parser.parse_args()

    # Output must live under held_out; never write sealed rows into processed/
    if "blind/held_out" not in str(args.out).replace("\\", "/"):
        raise SystemExit(f"Refusing to write sealed eval outside held_out/: {args.out}")

    rows = load_otrf_scenarios(args.scenarios, stages=[args.stage])
    kept: list[dict] = []
    dropped = 0
    for ex in rows:
        assert ex["meta"].get(HELD_OUT_META_FLAG) is True
        errs = verify_example(ex["messages"][1]["content"], ex["messages"][2]["content"])
        if errs:
            print(f"DROP {ex['id']}: {errs}")
            dropped += 1
            continue
        kept.append(ex)

    for ex in kept:
        assert ex["meta"]["split"] == "held_out_never_train"
        assert ex["meta"].get(HELD_OUT_META_FLAG) is True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"froze {len(kept)} sealed blind examples (dropped {dropped}) → {args.out}")
    print("POLICY: never pass this path to train_sft / train_shards / compose --out for training.")


if __name__ == "__main__":
    main()
