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
import time
from typing import Any

from eval_bench.adapters.base import ModelAdapter
from eval_bench.latency import TimedResult


class _StepClock:
    """A stopping criterion that never stops: generate() calls it once per
    new token, so its call times are the token times. It synchronizes CUDA
    first -- kernels launch asynchronously, and without the sync a timestamp
    could precede the token it claims to time. (generate() already syncs
    every step to check for finished sequences, so this adds no stall.)"""

    def __init__(self):
        self.times: list[float] = []

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times.append(time.perf_counter())
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


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

    def _prepare(self, query: str, tool_set: list[dict[str, Any]]):
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

        # top_k/min_p are passed explicitly rather than inherited from the
        # model's generation_config, matching openai_compatible.py's "send
        # every knob" rule. Greedy (temperature == 0) sends do_sample=False
        # only: the sampling knobs are inert then, and HF rejects
        # temperature=0.0 combined with sampling.
        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.sampling_params["max_tokens"]}
        if self.sampling_params["temperature"] > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=self.sampling_params["temperature"],
                top_p=self.sampling_params["top_p"],
                top_k=self.sampling_params["top_k"],
                min_p=self.sampling_params["min_p"],
            )
        else:
            gen_kwargs["do_sample"] = False
        seed = self.sampling_params.get("seed")
        if seed is not None:
            import torch
            torch.manual_seed(seed)
        return inputs, gen_kwargs

    def _decode_new(self, inputs, output_ids) -> str:
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)

    def generate(self, query: str, tool_set: list[dict[str, Any]]) -> str:
        inputs, gen_kwargs = self._prepare(query, tool_set)
        return self._decode_new(inputs, self.model.generate(**inputs, **gen_kwargs))

    def generate_timed(self, query: str, tool_set: list[dict[str, Any]],
                       timeout: float) -> TimedResult:
        """The latency pass's request (concurrency 1 only). TTFT runs from
        before the chat template is rendered -- a server's TTFT includes its
        own templating and tokenization -- to the first generated token.
        `timeout` is not enforced in-process; a request past it is still
        counted as a timeout."""
        from transformers import StoppingCriteriaList

        started = time.perf_counter()
        inputs, gen_kwargs = self._prepare(query, tool_set)
        clock = _StepClock()
        output_ids = self.model.generate(**inputs, **gen_kwargs,
                                         stopping_criteria=StoppingCriteriaList([clock]))
        total = time.perf_counter() - started
        text = self._decode_new(inputs, output_ids)
        if total > timeout:
            return TimedResult(status="timeout", total_s=total)
        ttft = clock.times[0] - started if clock.times else None
        return TimedResult(
            status="ok", total_s=total, ttft_s=ttft, first_delta_s=ttft,
            output_tokens=len(clock.times), prompt_tokens=inputs["input_ids"].shape[1],
            cached_tokens=None, content=text,
        )
