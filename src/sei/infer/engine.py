"""vLLM / transformers inference with optional guided JSON + RAG."""

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
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, [f"invalid_json: {e}"]
    errors = verify_example(telemetry_user_text, obj)
    return obj, errors


class SEIInferencer:
    """Lazy-loaded inferencer. Falls back to stub if model unavailable."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        backend: str = "transformers",
        use_guided: bool = True,
        stub: bool = False,
    ):
        self.model_path = model_path
        self.backend = backend
        self.use_guided = use_guided
        self.stub = stub or not model_path
        self._llm = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self.stub or self._llm is not None:
            return
        if self.backend == "vllm":
            from vllm import LLM
            self._llm = LLM(model=self.model_path, max_model_len=4096)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            from sei.train.chat_template import LLAMA3_CHAT_TEMPLATE

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
            if not self._tokenizer.chat_template:
                self._tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
            self._llm = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

    def generate(
        self,
        events: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        *,
        use_rag: bool = True,
    ) -> dict[str, Any]:
        messages = build_chat_messages(events, context, use_rag=use_rag)
        user_text = messages[1]["content"]

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
            params = SamplingParams(temperature=0.0, max_tokens=1024, **kwargs)
            outputs = self._llm.generate([prompt], params)
            text = outputs[0].outputs[0].text
        else:
            import torch

            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._llm.device)
            out = self._llm.generate(**inputs, max_new_tokens=1024, do_sample=False)
            text = self._tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )

        obj, errors = parse_and_verify(user_text, text)
        return {"sei": obj, "errors": errors, "raw": text, "stub": False}
