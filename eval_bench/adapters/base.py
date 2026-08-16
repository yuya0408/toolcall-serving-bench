"""Adapter layer: absorb per-model invocation differences.

Each adapter takes (query, tool_set, few_shot) and returns raw model output
text. Nothing here knows about scoring or decoding. Add one subclass per
backend (Ollama, PrismML, cloud). Article 1 compares models by swapping
adapters under an identical dataset/decoder/scorer.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def generate(self, query: str, tool_set: list[dict[str, Any]],
                 few_shot: list[dict[str, Any]]) -> str:
        """Return the model's raw text output for one query."""
        ...


class EchoAdapter(ModelAdapter):
    """Placeholder adapter for wiring/tests. Replace with real backends."""
    name = "echo-stub"

    def generate(self, query, tool_set, few_shot) -> str:
        return "{}"
