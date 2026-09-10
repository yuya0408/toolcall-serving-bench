"""Adapter layer: absorb per-model invocation differences.

Each adapter takes (query, tool_set) and returns the model's raw output.
Nothing here knows about scoring; parsing that output into a Prediction is
the paired Decoder's job (eval_bench/decoders/base.py), not the adapter's --
an adapter invokes the engine's API and hands back what it said, unparsed.
Add one subclass per backend. Article 1 compares models by swapping adapters
under an identical dataset/decoder/scorer.

No few_shot parameter: the harness uses each engine's chat/tools API
(docs/engine-performance-plan.md's '比較可能性の規律' -- 2026-09-10 switch from
raw completion endpoints), and Qwen3-8B is natively tool-calling-tuned, so
no in-prompt examples are needed to induce structured output the way they
would be for a raw-completion pattern-continuation approach.

What "raw output" means differs by what the paired Decoder expects:
  - transformers_hf: the generated text as-is (may contain a Hermes-style
    '<tool_call>...</tool_call>' block) -- see HermesTagDecoder.
  - vllm / llama_cpp: the JSON-serialized `message` object from the chat
    completions response (tool_calls or plain content) -- see
    OpenAIToolCallDecoder.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def generate(self, query: str, tool_set: list[dict[str, Any]]) -> str:
        """Return the model's raw output for one query (see module docstring
        for what "raw" means for this adapter's backend)."""
        ...


class EchoAdapter(ModelAdapter):
    """Placeholder adapter for wiring/tests. Replace with real backends."""
    name = "echo-stub"

    def generate(self, query, tool_set) -> str:
        return "{}"
