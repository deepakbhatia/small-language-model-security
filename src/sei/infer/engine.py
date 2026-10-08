"""vLLM / transformers inference with optional LoRA adapter + guided JSON + RAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sei import SYSTEM_PROMPT
from sei.normalize.ecs import render_telemetry_block
from sei.rag.attack_rag import render_rag_block
from sei.verify.validator import verify_example

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "sei_v1.json"
DEFAULT_BASE = "fdtn-ai/Foundation-Sec-8B"
DEFAULT_ADAPTER = REPO_ROOT / "checkpoints" / "sei-sft-stage1" / "adapter"


def build_prompt(events: list[dict[str, Any]], context: dict[str, Any] | None = None, *, use_rag: bool = True) -> str:
    user = render_telemetry_block(events, context)
    if use_rag:
        rag = render_rag_block(user)
        if rag:
            user = user + "\nATT&CK_CONTEXT:\n" + rag
    # Plain prompt text for engines without chat template applied yet
    return SYSTEM_PROMPT + "\n\n" + user + "\n\nSEI_JSON:"


def build_chat_messages(
    events: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
    *,
    use_rag: bool = True,
) -> list[dict[str, str]]:
    user = render_telemetry_block(events, context)
    if use_rag:
        rag = render_rag_block(user)
        if rag:
            user = user + "\nATT&CK_CONTEXT:\n" + rag
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_and_verify(telemetry_user_text: str, generated: str) -> tuple[dict[str, Any] | None, list[str]]:
    text = generated.strip()
    # Strip optional markdown fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    # Models often emit one valid object then continue; take the first JSON value.
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            obj, _end = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError as e:
            return None, [f"invalid_json: {e}"]
    if not isinstance(obj, dict):
        return None, ["invalid_json: expected object"]
    errors = verify_example(telemetry_user_text, obj)
    return obj, errors


def _resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SEIInferencer:
    """Lazy-loaded inferencer. Falls back to stub if model unavailable."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        adapter_path: str | Path | None = None,
        backend: str = "transformers",
        use_guided: bool = True,
        stub: bool = False,
        load_in_4bit: bool | None = None,
    ):
        self.model_path = model_path or DEFAULT_BASE
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.backend = backend
        self.use_guided = use_guided
        # Explicit stub=True wins; otherwise stub only when neither base nor adapter is usable.
        if stub:
            self.stub = True
        else:
            self.stub = False
        if load_in_4bit is None:
            import torch

            load_in_4bit = torch.cuda.is_available()
        self.load_in_4bit = bool(load_in_4bit)
        self._llm = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self.stub or self._llm is not None:
            return
        if self.backend == "vllm":
            if self.adapter_path:
                raise ValueError(
                    "vLLM backend does not load LoRA adapters here; "
                    "merge adapters first or use backend=transformers"
                )
            from vllm import LLM

            self._llm = LLM(model=self.model_path, max_model_len=4096)
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        from peft import PeftModel
        from sei.train.chat_template import LLAMA3_CHAT_TEMPLATE

        tok_src = str(self.adapter_path) if self.adapter_path and (self.adapter_path / "tokenizer.json").exists() else self.model_path
        self._tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
        if not self._tokenizer.chat_template:
            self._tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, Any] = {}
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        else:
            dtype = torch.bfloat16 if _resolve_device() != "cpu" else torch.float32
            load_kwargs["torch_dtype"] = dtype
            load_kwargs["device_map"] = "auto" if _resolve_device() == "cuda" else None

        print(f"[infer] loading base {self.model_path} (4bit={self.load_in_4bit})")
        base = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        if load_kwargs.get("device_map") is None:
            base = base.to(_resolve_device())

        if self.adapter_path:
            print(f"[infer] loading adapter {self.adapter_path}")
            self._llm = PeftModel.from_pretrained(base, str(self.adapter_path))
        else:
            self._llm = base
        self._llm.eval()

    def _model_device(self):
        return next(self._llm.parameters()).device

    def generate(
        self,
        events: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        *,
        use_rag: bool = True,
        max_new_tokens: int = 1024,
    ) -> dict[str, Any]:
        messages = build_chat_messages(events, context, use_rag=use_rag)
        return self.generate_chat(messages, max_new_tokens=max_new_tokens)

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Generate from chat messages (system + user). Used for golden eval."""
        user_text = next((m["content"] for m in messages if m["role"] == "user"), "")

        if self.stub:
            # Deterministic stub for demos / CI without GPU weights
            from sei.compose.synthetic import scenario_to_example
            from sei.compose.synthetic import SCENARIOS

            ex = scenario_to_example(SCENARIOS[0], stage=4)
            obj = json.loads(ex["messages"][-1]["content"])
            return {"sei": obj, "errors": verify_example(user_text, obj), "stub": True}

        self._ensure_loaded()
        if self.backend == "vllm":
            from vllm import SamplingParams
            from vllm.sampling_params import GuidedDecodingParams

            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) if self._tokenizer else (SYSTEM_PROMPT + "\n" + user_text)
            # For vLLM path without HF tokenizer chat — rebuild
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                from sei.train.chat_template import LLAMA3_CHAT_TEMPLATE

                tok = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
                if not tok.chat_template:
                    tok.chat_template = LLAMA3_CHAT_TEMPLATE
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            kwargs = {}
            if self.use_guided and SCHEMA_PATH.exists():
                kwargs["guided_decoding"] = GuidedDecodingParams(json=SCHEMA_PATH.read_text())
            params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, **kwargs)
            outputs = self._llm.generate([prompt], params)
            text = outputs[0].outputs[0].text
        else:
            import torch

            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model_device()) for k, v in inputs.items()}
            with torch.inference_mode():
                out = self._llm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            text = self._tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )

        obj, errors = parse_and_verify(user_text, text)
        return {
            "sei": obj,
            "errors": errors,
            "raw": text,
            "stub": False,
            "adapter": str(self.adapter_path) if self.adapter_path else None,
        }
