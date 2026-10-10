#!/usr/bin/env python3
"""Build a *soft* blind pack (GUIDE offset + late Atomic).

For the **sealed never-train** set use instead:
  python scripts/freeze_held_out_eval.py
  python scripts/eval_model.py --gold data/blind/held_out/eval.jsonl ...

This script is optional extra OOD signal only — it is NOT the sealed contract.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose blind SEI eval JSONL")
    parser.add_argument("--out", type=Path, default=Path("data/processed/blind"))
    parser.add_argument("--guide", type=Path, default=Path("data/raw/guide/GUIDE_Train.csv"))
    parser.add_argument("--guide-offset", type=int, default=500_000)
    parser.add_argument("--guide-limit", type=int, default=150)
    parser.add_argument("--atomic-dir", type=Path, default=Path("data/raw/atomic/atomics"))
    parser.add_argument("--atomic-min", default="T1200", help="Only Atomic techniques >= this id")
    parser.add_argument("--atomic-limit", type=int, default=300)
    parser.add_argument("--blind-yaml", type=Path, default=Path("data/blind/scenarios.yaml"))
    parser.add_argument("--attack-stages", default="4", help="Use stage 4 for full SEI labels")
    parser.add_argument("--skip-guide", action="store_true")
    parser.add_argument("--skip-atomic", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "scripts/compose_all.py",
        "--out",
        str(args.out),
        "--no-seed",
        "--otrf",
        str(args.blind_yaml),
        "--attack-stages",
        args.attack_stages,
        "--val-ratio",
        "0.0",
        "--val-monitor-size",
        "0",
    ]
    if not args.skip_atomic:
        cmd += [
            "--atomic-dir",
            str(args.atomic_dir),
            "--atomic-limit",
            str(args.atomic_limit),
            "--atomic-min-technique",
            args.atomic_min,
        ]
    if not args.skip_guide and args.guide.exists():
        cmd += [
            "--guide",
            str(args.guide),
            "--guide-offset",
            str(args.guide_offset),
            "--guide-limit",
            str(args.guide_limit),
            "--guide-stage",
            "2",
        ]
    else:
        print("GUIDE skipped or missing — Atomic + hand blind only")

    print(" ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)
    all_path = args.out / "all.jsonl"
    n = sum(1 for _ in all_path.open()) if all_path.exists() else 0
    print(f"\nBlind pack ready: {n} examples → {all_path}")
    print(
        "Eval with:\n"
        f"  python scripts/eval_model.py --adapter checkpoints/sei-sft-attack/adapter \\\n"
        f"    --gold {all_path} --limit 100 \\\n"
        f"    --pred-dir eval/preds/blind --out eval/reports/blind.json"
    )


if __name__ == "__main__":
    main()
