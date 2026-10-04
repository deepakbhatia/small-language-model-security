#!/usr/bin/env python3
"""Distill / SFT edge SKU on Qwen2.5-1.5B-Instruct from teacher JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge distill to Qwen2.5-1.5B-Instruct")
    parser.add_argument("--config", type=Path, default=Path("configs/distill_1p5b.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    # Reuse SFT entrypoint with student base model
    from sei.train.sft import main as sft_main

    # Write a temp override by mutating config path content via argv
    # Prefer calling sft with the distill config directly
    argv = ["--config", str(args.config)]
    if args.dry_run or cfg.get("dry_run"):
        argv.append("--dry-run")
    sft_main(argv)


if __name__ == "__main__":
    main()
