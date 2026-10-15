#!/usr/bin/env python3
"""Merge LoRA adapters into base weights (ship path).

Default adapter is the ORPO disposition champion — not stage-1 or consolidate.

  python scripts/merge_adapters.py
  python scripts/merge_adapters.py \\
    --adapter checkpoints/sei-orpo-disposition/adapter \\
    --out checkpoints/sei-8b-instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_ADAPTER = Path("checkpoints/sei-orpo-disposition/adapter")
DEFAULT_OUT = Path("checkpoints/sei-8b-instruct")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge SEI LoRA adapter into base HF weights")
    parser.add_argument("--base", default="fdtn-ai/Foundation-Sec-8B")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=DEFAULT_ADAPTER,
        help=f"LoRA adapter dir (default: {DEFAULT_ADAPTER})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Merged HF model dir (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    if not args.adapter.exists():
        raise SystemExit(
            f"adapter not found: {args.adapter}\n"
            "Sync the ship candidate first:\n"
            "  aws s3 sync s3://$BUCKET/sei-adapter-orpo/ "
            "checkpoints/sei-orpo-disposition/adapter/"
        )
    adapter_cfg = args.adapter / "adapter_config.json"
    if not adapter_cfg.exists():
        raise SystemExit(f"missing {adapter_cfg} — not a PEFT adapter directory")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sei.train.chat_template import LLAMA3_CHAT_TEMPLATE

    print(f"loading base {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    print(f"merging adapter {args.adapter}")
    merged = PeftModel.from_pretrained(base, str(args.adapter))
    merged = merged.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out))
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if not tok.chat_template:
        tok.chat_template = LLAMA3_CHAT_TEMPLATE
    tok.save_pretrained(str(args.out))
    print(f"merged → {args.out}")
    print("Next: python scripts/export_gguf.py --model", args.out, "--quant Q4_K_M")


if __name__ == "__main__":
    main()
