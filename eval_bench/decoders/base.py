"""Decoder layer.

This is the deliberate swap point (see the cost-asymmetry principle).

  - Article 1 uses IdentityDecoder: the model's raw output is parsed as-is.
    We measure the *bare* small-model ability, with no constraint help.

  - Article 3 will drop in a constrained decoder (e.g. GBNF grammar) HERE,
    touching nothing else in the harness. The delta between IdentityDecoder
    and a constrained decoder is precisely the "constrained decoding" step of
    the few-shot -> constrained-decoding -> FT comparison.

Keep this interface tiny on purpose: a decoder only turns raw model text into
a (tool, args) Prediction. It must not call an LLM or touch scoring.
"""
from __future__ import annotations
import json
import re
from abc import ABC, abstractmethod
from typing import Any

from eval_bench.scorers.separated import Prediction


class Decoder(ABC):
    @abstractmethod
    def decode(self, raw_output: str, tool_schema: dict[str, Any]) -> Prediction:
        ...


class IdentityDecoder(Decoder):
    """Article-1 decoder: parse the model's own JSON/tool-call output, no constraints."""

    def decode(self, raw_output: str, tool_schema: dict[str, Any]) -> Prediction:
        try:
            obj = json.loads(raw_output)
        except json.JSONDecodeError:
            # A parse failure is a real, reportable failure mode for small models:
            # it surfaces as tool=None (no valid call), which the scorer counts.
            return Prediction(tool=None, args={})
        tool = obj.get("tool") or obj.get("name")
        args = obj.get("args") or obj.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        return Prediction(tool=tool, args=args)


_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class HermesTagDecoder(IdentityDecoder):
    """transformers_hf decoder (engine-performance-plan's naive baseline has no
    serving-level tool-call parser): Qwen3 emits a tool call as a raw
    '<tool_call>{...}</tool_call>' text block rather than a structured API
    field. Only the first block is read -- Prediction supports one tool call
    per turn, matching this project's scoring design (see scorers/separated.py).

    No tag at all means the model answered in plain text, i.e. declined to
    call a tool -- not a parse failure, same as IdentityDecoder's tool=None
    for a well-formed non-call. Once the tag is stripped, the inner text is
    the same '{"name": ..., "arguments": ...}' shape IdentityDecoder already
    parses, so this class only strips the tag and reuses that logic rather
    than duplicating it.
    """

    def decode(self, raw_output: str, tool_schema: dict[str, Any]) -> Prediction:
        match = _TOOL_CALL_TAG_RE.search(raw_output)
        if match is None:
            return Prediction(tool=None, args={})
        return super().decode(match.group(1).strip(), tool_schema)


class OpenAIToolCallDecoder(Decoder):
    """vllm/llama_cpp decoder: both serve an OpenAI-compatible chat completions
    endpoint with a built-in tool-call parser (vLLM's --tool-call-parser
    hermes, llama.cpp server's --jinja-driven equivalent), so what the adapter
    hands back is already a structured `message` object (JSON-serialized),
    not free text -- the server has already done the tag-stripping that
    HermesTagDecoder does by hand for transformers_hf.

    OpenAI's wire format double-encodes: `message["tool_calls"][i]["function"]
    ["arguments"]` is itself a JSON *string*, not a nested object (unlike
    Hermes's raw text form, which is a single JSON blob with no extra
    encoding layer). Only the first tool_calls entry is read, same
    single-call assumption as HermesTagDecoder.

    A tool name that parses but carries malformed/missing arguments keeps
    tool=name with args={} rather than collapsing to tool=None: that failure
    is an argument-extraction problem, not a selection problem, and
    score_case's separated scoring (scorers/separated.py) depends on the two
    never being conflated.
    """

    def decode(self, raw_output: str, tool_schema: dict[str, Any]) -> Prediction:
        try:
            message = json.loads(raw_output)
        except json.JSONDecodeError:
            return Prediction(tool=None, args={})

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return Prediction(tool=None, args={})

        function = tool_calls[0].get("function", {})
        name = function.get("name")
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        return Prediction(tool=name, args=args)
