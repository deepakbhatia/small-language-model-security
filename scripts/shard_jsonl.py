#!/usr/bin/env python3
"""Split a JSONL train file into fixed-size shards for iterative training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=2000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    count = 0
    out = None
    total = 0

    with args.input.open() as f:
        for line in f:
            if not line.strip():
                continue
            if count == 0:
                if out:
                    out.close()
                path = args.out_dir / f"train_shard{shard_idx:02d}.jsonl"
                out = path.open("w")
                print(f"writing {path}")
            assert out is not None
            # validate json
            json.loads(line)
            out.write(line if line.endswith("\n") else line + "\n")
            count += 1
            total += 1
            if count >= args.shard_size:
                count = 0
                shard_idx += 1
    if out:
        out.close()
    print(f"done: {total} rows → {shard_idx + (1 if count else 0)} shard files in {args.out_dir}")


if __name__ == "__main__":
    main()
