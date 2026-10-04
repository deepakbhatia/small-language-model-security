#!/usr/bin/env python3
"""Download the SEI base model with resilient Hugging Face settings.

Workaround for XET/CAS decode errors:
  HF_HUB_DISABLE_XET=1 python scripts/download_base_model.py
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fdtn-ai/Foundation-Sec-8B")
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional local folder (else uses HF cache)",
    )
    parser.add_argument(
        "--disable-xet",
        action="store_true",
        default=True,
        help="Disable HF XET transport (default: on; fixes CAS decode errors)",
    )
    args = parser.parse_args()

    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=args.model,
        local_dir=args.local_dir,
        resume_download=True,
        max_workers=2,
    )
    print(f"downloaded → {path}")


if __name__ == "__main__":
    main()
