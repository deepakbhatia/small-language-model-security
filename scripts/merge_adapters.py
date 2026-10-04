#!/usr/bin/env python3
"""Merge LoRA adapters into base weights."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="fdtn-ai/Foundation-Sec-8B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sei.train.chat_template import LLAMA3_CHAT_TEMPLATE

    print(f"loading base {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, str(args.adapter))
    merged = merged.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out))
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if not tok.chat_template:
        tok.chat_template = LLAMA3_CHAT_TEMPLATE
    tok.save_pretrained(str(args.out))
    print(f"merged → {args.out}")


if __name__ == "__main__":
    main()
