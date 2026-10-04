#!/usr/bin/env python3
"""Export HuggingFace model to GGUF via llama.cpp convert+quantize (if available)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--llama-cpp", type=Path, default=None, help="Path to llama.cpp repo")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/gguf"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    f16 = args.out_dir / f"{args.model.name}-f16.gguf"
    qout = args.out_dir / f"{args.model.name}-{args.quant.lower()}.gguf"

    # Bundle grammar + schema alongside
    for rel in ["schemas/sei_v1.json", "grammars/sei_v1.gbnf", "data/attack/v16.1/technique_ids.txt", "NOTICE"]:
        src = Path(rel)
        if src.exists():
            dst = args.out_dir / src.name
            shutil.copy2(src, dst)

    manifest = args.out_dir / "airgap_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"model_dir={args.model}",
                f"quant={args.quant}",
                f"gguf={qout}",
                "grammar=sei_v1.gbnf",
                "schema=sei_v1.json",
                "attack_ids=technique_ids.txt",
                "serve_example=llama-cli -m MODEL.gguf --grammar-file sei_v1.gbnf -p PROMPT",
            ]
        )
        + "\n"
    )

    if args.llama_cpp is None:
        print(
            "llama.cpp path not provided — wrote air-gap manifest only.\n"
            "To convert:\n"
            "  python llama.cpp/convert_hf_to_gguf.py MODEL --outfile F16.gguf\n"
            "  llama-quantize F16.gguf QOUT " + args.quant
        )
        print(f"manifest → {manifest}")
        return

    convert = args.llama_cpp / "convert_hf_to_gguf.py"
    quantize = args.llama_cpp / "llama-quantize"
    if not convert.exists():
        raise SystemExit(f"missing {convert}")
    subprocess.check_call(["python", str(convert), str(args.model), "--outfile", str(f16)])
    if quantize.exists():
        subprocess.check_call([str(quantize), str(f16), str(qout), args.quant])
        print(f"quantized → {qout}")
    else:
        print(f"f16 gguf → {f16} (quantize binary not found)")


if __name__ == "__main__":
    main()
