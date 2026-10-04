#!/usr/bin/env python3
"""Eval harness CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sei.eval_metrics import evaluate_dirs, evaluate_paired, _load_sei_from_obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.gold.is_dir():
        report = evaluate_dirs(args.gold, args.pred)
    else:
        golds = [_load_sei_from_obj(json.loads(l)) for l in args.gold.read_text().splitlines() if l.strip()]
        preds = [_load_sei_from_obj(json.loads(l)) for l in args.pred.read_text().splitlines() if l.strip()]
        report = evaluate_paired(golds, preds)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
