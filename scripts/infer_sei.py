#!/usr/bin/env python3
"""Run SEI inference with base model + LoRA adapter.

Examples:
  # Default: checkpoints/sei-sft-stage1/adapter + Foundation-Sec-8B
  python scripts/infer_sei.py

  # Custom message
  python scripts/infer_sei.py --message 'cmd.exe /c whoami' --source sysmon

  # From a golden / chat JSONL row file
  python scripts/infer_sei.py --from-json data/golden/01_cmd.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sei.infer.engine import DEFAULT_ADAPTER, DEFAULT_BASE, SEIInferencer


def _events_from_args(args: argparse.Namespace) -> tuple[list[dict], dict | None]:
    if args.from_json:
        row = json.loads(Path(args.from_json).read_text())
        if "messages" in row:
            # Prefer re-parsing telemetry from user turn is hard; use --message fallback fields
            # if present, else a minimal event from user content.
            user = row["messages"][1]["content"]
            return (
                [
                    {
                        "timestamp": "2024-01-01T00:00:00Z",
                        "source": "other",
                        "raw": {},
                        "message": user[:2000],
                    }
                ],
                None,
            )
        if "events" in row:
            return row["events"], row.get("context")
        raise SystemExit(f"unrecognized JSON shape in {args.from_json}")

    events = [
        {
            "timestamp": "2024-06-01T12:00:00Z",
            "source": args.source,
            "host": args.host,
            "raw": {"CommandLine": args.message} if args.source == "sysmon" else {},
            "message": args.message,
        }
    ]
    context = {"asset_criticality": args.criticality} if args.criticality else None
    return events, context


def main() -> None:
    parser = argparse.ArgumentParser(description="SEI adapter inference")
    parser.add_argument("--base", default=os.environ.get("SEI_MODEL_PATH", DEFAULT_BASE))
    parser.add_argument(
        "--adapter",
        default=os.environ.get("SEI_ADAPTER_PATH", str(DEFAULT_ADAPTER)),
        help="LoRA adapter dir (default: checkpoints/sei-sft-stage1/adapter)",
    )
    parser.add_argument("--no-adapter", action="store_true", help="Base model only")
    parser.add_argument("--stub", action="store_true", help="No weights (CI/demo)")
    parser.add_argument("--load-4bit", action="store_true", help="Force 4-bit (CUDA)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--message", default="cmd.exe /c whoami /all")
    parser.add_argument("--source", default="sysmon")
    parser.add_argument("--host", default="workstation-01")
    parser.add_argument("--criticality", default="high")
    parser.add_argument("--from-json", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=None, help="Write full result JSON")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    adapter = None if args.no_adapter or args.stub else args.adapter
    if adapter and not Path(adapter).exists():
        raise SystemExit(f"adapter not found: {adapter}")

    load_in_4bit = None
    if args.load_4bit:
        load_in_4bit = True
    elif args.no_4bit:
        load_in_4bit = False

    eng = SEIInferencer(
        model_path=args.base,
        adapter_path=adapter,
        stub=args.stub,
        load_in_4bit=load_in_4bit,
    )
    events, context = _events_from_args(args)
    result = eng.generate(
        events,
        context,
        use_rag=not args.no_rag,
        max_new_tokens=args.max_new_tokens,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")
    print(json.dumps({k: result[k] for k in ("sei", "errors", "stub", "adapter") if k in result}, indent=2))
    if result.get("sei") is None and result.get("raw"):
        print("\n--- raw ---\n" + result["raw"][:2000])


if __name__ == "__main__":
    main()
