#!/usr/bin/env python3
"""Verify lm-format-enforcer + sei_v1 schema are ready for ship/eval.

Exit 0 when transformers guided decode can be required; else print install hint and exit 1.

  pip install -e ".[train]"
  python scripts/check_guided.py
"""

from __future__ import annotations

from sei.infer.constrained import (
    INSTALL_HINT,
    SCHEMA_PATH,
    GuidedDecodeUnavailable,
    assert_guided_ready,
    guided_available_transformers,
)


def main() -> None:
    try:
        assert_guided_ready(backend="transformers")
    except GuidedDecodeUnavailable as e:
        print(f"FAIL: {e}")
        raise SystemExit(1) from e
    print(f"OK: schema={SCHEMA_PATH}")
    print(f"OK: lm-format-enforcer importable={guided_available_transformers()}")
    print("Ship eval: python scripts/eval_model.py ... --require-guided")


if __name__ == "__main__":
    main()
