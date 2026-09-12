"""transformers_hf adapter: the naive baseline, in-process via transformers'
own generate() with no serving-level optimizations (see
docs/engine-performance-plan.md's engine roster).

apply_chat_template(..., tools=tool_set) renders the same Hermes-aware
prompt format Qwen3-8B was tuned on, but transformers has no server-side
tool-call parser to strip the result -- the model still answers with a raw
'<tool_call>{...}</tool_call>' text block, which HermesTagDecoder
(decoders/base.py) is responsible for parsing. This adapter returns that
text as-is, unparsed, per the ModelAdapter contract.

The constructor takes an already-loaded tokenizer/model rather than a
model_id, so tests can inject fakes without installing torch/transformers.
Real callers should go through from_pretrained().
"""
from __future__ import annotations
from typing import Any

from eval_bench.adapters.base import ModelAdapter


class TransformersHFAdapter(ModelAdapter):
    name = "transformers_hf"

    def __init__(self, tokenizer, model, sampling_params: dict[str, Any]):
        self.tokenizer = tokenizer
        self.model = model
        self.sampling_params = sampling_params

    @classmethod
    def from_pretrained(cls, model_id: str, sampling_params: dict[str, Any],
                         device: str = "cuda") -> "TransformersHFAdapter":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # g6.xlarge has 16GB host RAM vs. ~16GB of bf16 weights: a plain
        # from_pretrained().to(device) materializes the model on the host
        # before moving it, which OOMs. device_map loads weights straight
        # onto the target device instead.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map={"": device},
        )
        return cls(tokenizer, model, sampling_params)

    def generate(self, query: str, tool_set: list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": query}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tool_set,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # docs/engine-performance-plan.md: measure selection/arg
                                     # extraction, not reasoning-trace quality or its latency cost
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.sampling_params["max_tokens"],
            "temperature": self.sampling_params["temperature"],
            "top_p": self.sampling_params["top_p"],
            "do_sample": self.sampling_params["temperature"] > 0,
        }
        seed = self.sampling_params.get("seed")
        if seed is not None:
            import torch
            torch.manual_seed(seed)

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][prompt_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
