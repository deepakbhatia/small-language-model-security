#!/usr/bin/env python3
"""Train over JSONL shards with resume + optional S3 sync after each shard."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", type=Path, default=Path("data/processed/v1/shards"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/sft_stage1.yaml"))
    parser.add_argument("--run-config", type=Path, default=Path("configs/sft_stage1_aws.yaml"))
    parser.add_argument("--val-file", default="data/processed/v1/val.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/sei-sft-stage1")
    parser.add_argument("--adapter-s3", default="", help="s3://bucket/prefix for adapter sync")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Cap steps (smoke)")
    parser.add_argument("--eval-strategy", default="steps")
    parser.add_argument("--eval-steps", type=int, default=500)
    args = parser.parse_args()

    shards = sorted(args.shards_dir.glob("train_shard*.jsonl"))
    if not shards:
        raise SystemExit(f"No shards in {args.shards_dir}")
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    shards = shards[args.start_shard :]
    adapter = Path(args.output_dir) / "adapter"

    for i, shard in enumerate(shards):
        global_i = args.start_shard + i
        cfg = yaml.safe_load(args.base_config.read_text())
        cfg.update(
            {
                "train_file": str(shard),
                "val_file": args.val_file,
                "output_dir": args.output_dir,
                "max_length": args.max_length,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": 1.0e-4 if global_i == 0 else 5.0e-5,
                "eval_strategy": args.eval_strategy,
                "eval_steps": args.eval_steps,
                "save_steps": max(args.eval_steps, 50),
                "logging_steps": 5 if args.max_steps else 10,
                "per_device_eval_batch_size": 4,
            }
        )
        if args.max_steps is not None:
            cfg["max_steps"] = args.max_steps
            cfg["eval_strategy"] = "no"
        if adapter.exists() and global_i > 0:
            cfg["resume_adapter"] = str(adapter)
        else:
            cfg.pop("resume_adapter", None)

        args.run_config.write_text(yaml.safe_dump(cfg))
        print(f"\n=== {shard.name} ===\n{args.run_config.read_text()}")
        rc = subprocess.call(
            [sys.executable, "scripts/train_sft.py", "--config", str(args.run_config)]
        )
        if rc != 0:
            raise SystemExit(f"train failed on {shard.name} exit={rc}")
        if not adapter.exists():
            raise SystemExit(f"adapter missing after {shard.name}")
        if args.adapter_s3:
            sync = [
                "aws",
                "s3",
                "sync",
                str(adapter),
                args.adapter_s3.rstrip("/") + "/",
            ]
            print(" ".join(sync))
            subprocess.check_call(sync)
            print(f"synced adapter → {args.adapter_s3}")

    print("train_shards done")


if __name__ == "__main__":
    main()
