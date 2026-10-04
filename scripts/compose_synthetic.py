#!/usr/bin/env python3
"""Compose synthetic Atomic/Sigma seed JSONL for all curriculum stages."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from sei.compose.synthetic import iter_seed_examples
from sei.verify.validator import verify_example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/processed/seed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = iter_seed_examples([1, 2, 3, 4])
    kept = []
    for row in rows:
        user = row["messages"][1]["content"]
        errs = verify_example(user, row["messages"][2]["content"])
        if errs:
            raise SystemExit(f"verify failed for {row['id']}: {errs}")
        kept.append(row)

    # Split by split_group to avoid leakage
    groups = sorted({r["meta"]["split_group"] for r in kept})
    random.shuffle(groups)
    n_val = max(1, int(len(groups) * args.val_ratio))
    val_groups = set(groups[:n_val])
    train = [r for r in kept if r["meta"]["split_group"] not in val_groups]
    val = [r for r in kept if r["meta"]["split_group"] in val_groups]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, subset in [("train.jsonl", train), ("val.jsonl", val), ("all.jsonl", kept)]:
        path = args.out / name
        with path.open("w") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(subset)} → {path}")


if __name__ == "__main__":
    main()
