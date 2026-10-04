"""QLoRA SFT training entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def filter_stage(rows: list[dict], stage: int) -> list[dict]:
    return [r for r in rows if int(r.get("meta", {}).get("curriculum_stage", 4)) <= stage]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SEI QLoRA curriculum SFT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate data only; skip GPU train")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    stage = args.stage or int(cfg.get("stage", 4))
    train_path = Path(cfg["train_file"])
    val_path = Path(cfg.get("val_file") or cfg["train_file"])

    train_rows = filter_stage(load_jsonl(train_path), stage)
    val_rows = filter_stage(load_jsonl(val_path), stage)
    print(f"[sft] stage={stage} train={len(train_rows)} val={len(val_rows)}")

    if args.dry_run or cfg.get("dry_run"):
        out = Path(cfg.get("output_dir", "checkpoints/dry-run"))
        out.mkdir(parents=True, exist_ok=True)
        (out / "dry_run_manifest.json").write_text(
            json.dumps({"stage": stage, "train": len(train_rows), "val": len(val_rows)}, indent=2)
        )
        print(f"[sft] dry-run ok → {out}")
        return

    # Heavy imports only when actually training
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from sei.train.chat_template import LLAMA3_CHAT_TEMPLATE

    model_name = cfg.get("base_model", "fdtn-ai/Foundation-Sec-8B")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    # Prefer bf16 on CUDA Ampere+; fall back for Apple Silicon / older GPUs
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=int(cfg.get("lora_r", 32)),
        lora_alpha=int(cfg.get("lora_alpha", 64)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, lora)

    def formatting_func(row):
        return tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    # TRL 1.14+ SFTConfig dropped warmup_ratio — use warmup_steps
    batch_size = int(cfg.get("batch_size", 1))
    grad_accum = int(cfg.get("grad_accum", 16))
    epochs = float(cfg.get("epochs", 2))
    steps_per_epoch = max(1, len(train_rows) // max(1, batch_size * grad_accum))
    total_steps = max(1, int(steps_per_epoch * epochs))
    if "warmup_steps" in cfg:
        warmup_steps = int(cfg["warmup_steps"])
    else:
        warmup_steps = max(1, int(total_steps * float(cfg.get("warmup_ratio", 0.03))))

    sft_args = SFTConfig(
        output_dir=cfg.get("output_dir", f"checkpoints/sei-sft-stage{stage}"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(cfg.get("lr", 1e-4)),
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=int(cfg.get("logging_steps", 20)),
        eval_strategy="steps",
        eval_steps=int(cfg.get("eval_steps", 200)),
        save_steps=int(cfg.get("save_steps", 200)),
        bf16=use_bf16,
        fp16=use_fp16,
        max_length=int(cfg.get("max_length", 4096)),
        packing=False,
        gradient_checkpointing=True,
        report_to=cfg.get("report_to", "none"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    if adapter := cfg.get("resume_adapter"):
        print(f"[sft] note: resume_adapter={adapter} (load via peft before train if needed)")
    trainer.train()
    out = Path(cfg.get("output_dir", f"checkpoints/sei-sft-stage{stage}")) / "adapter"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[sft] saved adapter → {out}")


if __name__ == "__main__":
    main()
