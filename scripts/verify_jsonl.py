#!/usr/bin/env python3
"""Verify SEI JSONL training files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sei.verify.validator import verify_example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    ok = bad = 0
    shown = 0
    with args.input.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            user = row["messages"][1]["content"]
            asst = row["messages"][2]["content"]
            errs = verify_example(user, asst)
            if errs:
                bad += 1
                if shown < args.max_errors:
                    print(f"FAIL {row.get('id')}: {errs}")
                    shown += 1
            else:
                ok += 1
    print(f"ok={ok} bad={bad}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
