#!/usr/bin/env python3
"""Compose all available sources into processed JSONL (GUIDE + Atomic/Sigma/OTRF)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from sei.compose.atomic import load_atomic_dir
from sei.compose.guide import guide_row_to_example
from sei.compose.otrf import load_otrf_jsonl, load_otrf_scenarios
from sei.compose.sigma import load_sigma_dir
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


def _stages(arg: str, default: list[int]) -> list[int]:
    if arg.strip():
        return [int(x) for x in arg.split(",") if x.strip()]
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--guide", type=Path, default=None, help="Path to GUIDE CSV")
    parser.add_argument("--guide-limit", type=int, default=5000)
    parser.add_argument("--guide-offset", type=int, default=0)
    parser.add_argument("--guide-stage", type=int, default=1)
    parser.add_argument(
        "--seed-stages",
        default="",
        help="Synthetic seed stages (default: 1..guide-stage; empty with --no-seed to skip)",
    )
    parser.add_argument("--no-seed", action="store_true", help="Skip built-in synthetic seed scenarios")
    parser.add_argument(
        "--atomic-dir",
        type=Path,
        default=None,
        help="Atomic Red Team atomics/ directory (data/raw/atomic/atomics)",
    )
    parser.add_argument("--atomic-limit", type=int, default=None)
    parser.add_argument(
        "--sigma-dir",
        type=Path,
        default=None,
        help="Sigma rules directory (data/raw/sigma/rules)",
    )
    parser.add_argument("--sigma-limit", type=int, default=None)
    parser.add_argument(
        "--otrf",
        type=Path,
        default=Path("data/otrf/scenarios.yaml"),
        help="Curated OTRF-style scenarios YAML (set empty path to skip)",
    )
    parser.add_argument("--otrf-jsonl", type=Path, default=None, help="Optional extra OTRF JSONL")
    parser.add_argument("--otrf-limit", type=int, default=None)
    parser.add_argument(
        "--attack-stages",
        default="",
        help="Curriculum stages for Atomic/Sigma/OTRF (default: 2,3,4)",
    )
    parser.add_argument(
        "--attack-mix",
        action="store_true",
        help="Enable default Atomic+Sigma+OTRF paths if present under data/raw and data/otrf",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-monitor-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    attack_stages = _stages(args.attack_stages, [2, 3, 4])
    if args.no_seed:
        seed_stages: list[int] = []
        examples: list[dict] = []
    else:
        seed_stages = _stages(args.seed_stages, list(range(1, max(1, int(args.guide_stage)) + 1)))
        examples = iter_seed_examples(seed_stages)
        print(f"synthetic seed stages={seed_stages} n={len(examples)}")

    if args.guide and args.guide.exists():
        g = load_guide_csv(
            args.guide,
            limit=args.guide_limit,
            stage=args.guide_stage,
            offset=args.guide_offset,
        )
        examples.extend(g)
        print(
            f"loaded GUIDE n={len(g)} from {args.guide} "
            f"(offset={args.guide_offset} limit={args.guide_limit} stage={args.guide_stage})"
        )
    elif args.guide:
        print(f"GUIDE path missing: {args.guide}")

    atomic_dir = args.atomic_dir
    sigma_dir = args.sigma_dir
    if args.attack_mix:
        atomic_dir = atomic_dir or Path("data/raw/atomic/atomics")
        sigma_dir = sigma_dir or Path("data/raw/sigma/rules")

    if atomic_dir and Path(atomic_dir).exists():
        a = load_atomic_dir(Path(atomic_dir), stages=attack_stages, limit=args.atomic_limit)
        examples.extend(a)
        print(f"loaded Atomic n={len(a)} from {atomic_dir} stages={attack_stages}")
    elif atomic_dir:
        print(f"Atomic dir missing: {atomic_dir}")

    if sigma_dir and Path(sigma_dir).exists():
        s = load_sigma_dir(Path(sigma_dir), stages=attack_stages, limit=args.sigma_limit)
        examples.extend(s)
        print(f"loaded Sigma n={len(s)} from {sigma_dir} stages={attack_stages}")
    elif sigma_dir:
        print(f"Sigma dir missing: {sigma_dir}")

    otrf_path = args.otrf
    if otrf_path and str(otrf_path) not in {"", "-", "none"} and Path(otrf_path).exists():
        o = load_otrf_scenarios(Path(otrf_path), stages=attack_stages, limit=args.otrf_limit)
        examples.extend(o)
        print(f"loaded OTRF curated n={len(o)} from {otrf_path}")
    if args.otrf_jsonl and args.otrf_jsonl.exists():
        oj = load_otrf_jsonl(args.otrf_jsonl, stages=attack_stages, limit=args.otrf_limit)
        examples.extend(oj)
        print(f"loaded OTRF jsonl n={len(oj)} from {args.otrf_jsonl}")

    kept = []
    dropped = 0
    drop_reasons: dict[str, int] = {}
    for ex in examples:
        errs = verify_example(ex["messages"][1]["content"], ex["messages"][2]["content"])
        if errs:
            dropped += 1
            key = errs[0].split(":")[0] if errs else "unknown"
            drop_reasons[key] = drop_reasons.get(key, 0) + 1
            continue
        kept.append(ex)

    if not kept:
        raise SystemExit(f"no examples kept (dropped={dropped} reasons={drop_reasons})")

    train, val_full = split_by_group(kept, val_ratio=args.val_ratio, seed=args.seed)
    rng = random.Random(args.seed)
    val_shuffled = list(val_full)
    rng.shuffle(val_shuffled)
    monitor_n = max(0, min(int(args.val_monitor_size), len(val_shuffled)))
    val_monitor = val_shuffled[:monitor_n]

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "val.jsonl", val_monitor)
    write_jsonl(args.out / "val_final.jsonl", val_full)
    write_jsonl(args.out / "all.jsonl", kept)
    print(
        f"wrote train={len(train)} val(monitor)={len(val_monitor)} "
        f"val_final={len(val_full)} all={len(kept)} "
        f"(dropped {dropped} {drop_reasons}, val_ratio={args.val_ratio}) → {args.out}"
    )


if __name__ == "__main__":
    main()
