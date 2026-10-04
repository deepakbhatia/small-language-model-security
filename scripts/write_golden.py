#!/usr/bin/env python3
"""Materialize data/golden/*.json from synthetic scenarios."""

from __future__ import annotations

import json
from pathlib import Path

from sei.compose.synthetic import SCENARIOS, scenario_to_example


def main() -> None:
    out = Path("data/golden")
    out.mkdir(parents=True, exist_ok=True)
    for sc in SCENARIOS:
        ex = scenario_to_example(sc, stage=4)
        (out / f"{sc['id']}.json").write_text(json.dumps(ex, indent=2) + "\n")
    print(f"wrote {len(SCENARIOS)} golden examples → {out}")


if __name__ == "__main__":
    main()
