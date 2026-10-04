"""ORPO preference alignment entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from sei.train.preferences import RECIPE_BUILDERS, build_preference_row


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def expand_preferences(sft_rows: list[dict], recipes: list[str]) -> list[dict]:
    out = []
    for row in sft_rows:
        if int(row.get("meta", {}).get("curriculum_stage", 4)) < 4:
            continue
        gold = row["messages"][-1]["content"]
        for recipe in recipes:
            pref = build_preference_row(row["messages"], gold, recipe=recipe)
            out.append(pref)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SEI ORPO alignment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    recipes = cfg.get("recipes", list(RECIPE_BUILDERS.keys()))
    rows = load_jsonl(Path(cfg["train_file"]))
    prefs = expand_preferences(rows, recipes)
    print(f"[orpo] preference pairs={len(prefs)} recipes={recipes}")

    pref_out = Path(cfg.get("preference_file", "data/processed/orpo/preferences.jsonl"))
    pref_out.parent.mkdir(parents=True, exist_ok=True)
    with pref_out.open("w") as f:
        for p in prefs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[orpo] wrote {pref_out}")

    if args.dry_run or cfg.get("dry_run"):
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import ORPOConfig, ORPOTrainer

    from sei.train.chat_template import LLAMA3_CHAT_TEMPLATE

    model_name = cfg.get("base_model", "fdtn-ai/Foundation-Sec-8B")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

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
    if adapter := cfg.get("resume_adapter"):
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        lora = LoraConfig(
            r=int(cfg.get("lora_r", 32)),
            lora_alpha=int(cfg.get("lora_alpha", 64)),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora)

    def to_orpo(p: dict) -> dict:
        prompt = tokenizer.apply_chat_template(
            p["prompt_messages"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": p["chosen"], "rejected": p["rejected"]}

    ds = Dataset.from_list([to_orpo(p) for p in prefs])
    orpo_args = ORPOConfig(
        output_dir=cfg.get("output_dir", "checkpoints/sei-orpo"),
        per_device_train_batch_size=int(cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 16)),
        learning_rate=float(cfg.get("lr", 5e-6)),
        num_train_epochs=float(cfg.get("epochs", 1)),
        beta=float(cfg.get("beta", 0.1)),
        bf16=use_bf16,
        fp16=use_fp16,
        max_length=int(cfg.get("max_length", 4096)),
        max_completion_length=int(cfg.get("max_completion_length", 1024)),
        report_to=cfg.get("report_to", "none"),
    )
    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    out = Path(cfg.get("output_dir", "checkpoints/sei-orpo")) / "adapter"
    trainer.save_model(str(out))
    print(f"[orpo] saved → {out}")


if __name__ == "__main__":
    main()
