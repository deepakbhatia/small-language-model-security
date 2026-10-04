#!/usr/bin/env python3
"""Write public benchmark leaderboard stub from eval JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, default=Path("eval/reports"))
    parser.add_argument("--out", type=Path, default=Path("eval/LEADERBOARD.md"))
    args = parser.parse_args()

    lines = [
        "# SEI Public Benchmark Leaderboard",
        "",
        "Primary metrics: disposition macro-F1 (GUIDE), technique exact-ID F1, evidence span F1, severity ordinal MAE.",
        "",
        "| Run | Disposition macro-F1 | Technique F1 | Evidence F1 | Severity MAE |",
        "|-----|---------------------:|-------------:|------------:|-------------:|",
    ]
    if args.reports.exists():
        for path in sorted(args.reports.glob("*.json")):
            rep = json.loads(path.read_text())
            lines.append(
                "| {name} | {d:.3f} | {t:.3f} | {e:.3f} | {s:.3f} |".format(
                    name=path.stem,
                    d=rep.get("disposition", {}).get("macro_f1", 0.0),
                    t=rep.get("techniques", {}).get("exact_id_f1", 0.0),
                    e=rep.get("evidence", {}).get("f1", 0.0),
                    s=rep.get("severity", {}).get("ordinal_mae", 0.0),
                )
            )
    else:
        lines.append("| baseline-identity | — | — | — | — |")

    lines.append("")
    lines.append("Submit by opening a PR with `eval/reports/<run>.json` from `scripts/eval_harness.py`.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
