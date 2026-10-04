#!/usr/bin/env python3
"""CI smoke test against golden examples (verifier + stub infer)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sei.infer.engine import SEIInferencer
from sei.verify.validator import verify_example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=Path("data/golden"))
    args = parser.parse_args()

    files = sorted(args.golden.glob("*.json"))
    if not files:
        raise SystemExit(f"no golden files in {args.golden}")

    # Verifier path
    for gf in files:
        row = json.loads(gf.read_text())
        user = row["messages"][1]["content"]
        asst = row["messages"][2]["content"]
        errs = verify_example(user, asst)
        assert not errs, (gf.name, errs)

    # Stub inferencer returns schema-shaped output
    eng = SEIInferencer(stub=True)
    result = eng.generate(
        [{"timestamp": "2024-01-01T00:00:00Z", "source": "sysmon", "raw": {"CommandLine": "cmd.exe /c whoami"}, "message": "cmd.exe /c whoami"}],
        {"asset_criticality": "high"},
    )
    assert result["sei"] is not None
    print(f"smoke ok: {len(files)} golden files, stub infer ok")


if __name__ == "__main__":
    main()
