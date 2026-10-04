#!/usr/bin/env python3
"""Download Microsoft GUIDE dataset via Kaggle API (optional)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw/guide"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if shutil.which("kaggle") is None:
        print(
            "kaggle CLI not found. Install kaggle, place ~/.kaggle/kaggle.json, then:\n"
            "  kaggle datasets download -d Microsoft/microsoft-security-incident-prediction -p data/raw/guide --unzip"
        )
        raise SystemExit(2)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "Microsoft/microsoft-security-incident-prediction",
        "-p",
        str(args.out),
        "--unzip",
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print(f"GUIDE downloaded to {args.out}")


if __name__ == "__main__":
    main()
