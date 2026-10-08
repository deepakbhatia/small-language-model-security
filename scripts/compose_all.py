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


def load_guide_csv(
    path: Path,
    *,
    limit: int | None,
    stage: int,
    offset: int = 0,
) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < offset:
                continue
            if limit is not None and len(rows) >= limit:
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
    parser.add_argument(
        "--guide-offset",
        type=int,
        default=0,
        help="Skip first N GUIDE rows (e.g. 20000 to continue after an earlier run)",
    )
    parser.add_argument("--guide-stage", type=int, default=1)
    parser.add_argument(
        "--seed-stages",
        default="",
        help="Comma list of synthetic curriculum stages (default: 1..guide-stage)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of split_groups for val")
    parser.add_argument(
        "--val-monitor-size",
        type=int,
        default=200,
        help="Small val.jsonl size for per-shard checks; full holdout written to val_final.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed_stages.strip():
        seed_stages = [int(x) for x in args.seed_stages.split(",") if x.strip()]
    else:
        seed_stages = list(range(1, max(1, int(args.guide_stage)) + 1))
    examples = iter_seed_examples(seed_stages)
    if args.guide and args.guide.exists():
        examples.extend(
            load_guide_csv(
                args.guide,
                limit=args.guide_limit,
                stage=args.guide_stage,
                offset=args.guide_offset,
            )
        )
        print(
            f"loaded GUIDE from {args.guide} "
            f"(offset={args.guide_offset} limit={args.guide_limit} stage={args.guide_stage})"
        )
    else:
        print("GUIDE CSV not provided — synthetic only")
    print(f"synthetic seed stages={seed_stages}")

    kept = []
    dropped = 0
    for ex in examples:
        errs = verify_example(ex["messages"][1]["content"], ex["messages"][2]["content"])
        if errs:
            dropped += 1
            continue
        kept.append(ex)

    train, val_full = split_by_group(kept, val_ratio=args.val_ratio, seed=args.seed)

    rng = random.Random(args.seed)
    val_shuffled = list(val_full)
    rng.shuffle(val_shuffled)
    monitor_n = max(0, min(int(args.val_monitor_size), len(val_shuffled)))
    val_monitor = val_shuffled[:monitor_n]

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "val.jsonl", val_monitor)  # fast per-shard monitor
    write_jsonl(args.out / "val_final.jsonl", val_full)  # full holdout for last eval
    write_jsonl(args.out / "all.jsonl", kept)
    print(
        f"wrote train={len(train)} val(monitor)={len(val_monitor)} "
        f"val_final={len(val_full)} all={len(kept)} "
        f"(dropped {dropped}, val_ratio={args.val_ratio}) → {args.out}"
    )


if __name__ == "__main__":
    main()
