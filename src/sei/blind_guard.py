"""Hard guardrails: sealed blind held-out data must never enter training."""

from __future__ import annotations

from pathlib import Path

# Canonical sealed eval tree (git-tracked). Never use as train/val for SFT/ORPO.
HELD_OUT_DIRNAME = "held_out"
HELD_OUT_MARKERS = (
    f"data/blind/{HELD_OUT_DIRNAME}",
    f"data\\blind\\{HELD_OUT_DIRNAME}",
    "/blind/held_out/",
    "\\blind\\held_out\\",
)
HELD_OUT_META_FLAG = "blind_held_out"


class BlindHeldOutError(ValueError):
    """Raised when training would consume sealed blind data."""


def is_held_out_path(path: str | Path | None) -> bool:
    if path is None:
        return False
    text = str(path).replace("\\", "/")
    lowered = text.lower()
    if "/blind/held_out/" in lowered or lowered.rstrip("/").endswith("/blind/held_out"):
        return True
    # also catch relative forms
    return "data/blind/held_out" in lowered


def assert_not_held_out_path(path: str | Path | None, *, role: str = "path") -> None:
    if is_held_out_path(path):
        raise BlindHeldOutError(
            f"Refusing to use sealed blind held-out {role}: {path}. "
            "data/blind/held_out/ is eval-only and must never be used for training."
        )


def assert_row_not_held_out(row: dict, *, role: str = "example") -> None:
    meta = row.get("meta") or {}
    if meta.get(HELD_OUT_META_FLAG) is True or meta.get("split") == "held_out_never_train":
        raise BlindHeldOutError(
            f"Refusing to train on sealed blind {role} id={row.get('id')!r}. "
            f"meta.{HELD_OUT_META_FLAG}=true is eval-only."
        )


def assert_dataset_not_held_out(rows: list[dict], *, role: str = "dataset") -> None:
    for row in rows:
        assert_row_not_held_out(row, role=role)
