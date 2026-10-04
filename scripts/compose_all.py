#!/usr/bin/env python3
"""Compose all available sources into processed JSONL (GUIDE optional)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from sei.compose.guide import guide_row_to_example
from sei.compose.synthetic import iter_seed_examples
from sei.verify.validator import verify_example


def load_guide_csv(path: Path, *, limit: int | None, stage: int) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(guide_row_to_example(row, stage=stage, idx=i))
    return rows


def split_by_group(
    rows: list[dict],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Hold out by meta.split_group so GUIDE OrgId / scenario families don't leak."""
    groups = sorted({r.get("meta", {}).get("split_group", r.get("id", "unknown")) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_ratio))) if groups else 0
    # Keep at least one train group when possible
    if len(groups) > 1:
        n_val = min(n_val, len(groups) - 1)
    val_groups = set(groups[:n_val])
    train = [r for r in rows if r.get("meta", {}).get("split_group", r.get("id")) not in val_groups]
    val = [r for r in rows if r.get("meta", {}).get("split_group", r.get("id")) in val_groups]
    return train, val


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--guide", type=Path, default=None, help="Path to GUIDE CSV")
    parser.add_argument("--guide-limit", type=int, default=5000)
    parser.add_argument("--guide-stage", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of split_groups for val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = iter_seed_examples([1, 2, 3, 4])
    if args.guide and args.guide.exists():
        examples.extend(load_guide_csv(args.guide, limit=args.guide_limit, stage=args.guide_stage))
        print(f"loaded GUIDE from {args.guide}")
    else:
        print("GUIDE CSV not provided — synthetic only")

    kept = []
    dropped = 0
    for ex in examples:
        errs = verify_example(ex["messages"][1]["content"], ex["messages"][2]["content"])
        if errs:
            dropped += 1
            continue
        kept.append(ex)

    train, val = split_by_group(kept, val_ratio=args.val_ratio, seed=args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "val.jsonl", val)
    write_jsonl(args.out / "all.jsonl", kept)
    print(
        f"wrote train={len(train)} val={len(val)} all={len(kept)} "
        f"(dropped {dropped}, val_ratio={args.val_ratio}) → {args.out}"
    )


if __name__ == "__main__":
    main()
