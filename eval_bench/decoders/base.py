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
        import json
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
