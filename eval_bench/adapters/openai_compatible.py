"""vllm / llama_cpp adapters: both serve an OpenAI-compatible chat completions
endpoint (llama.cpp server's /v1/chat/completions, alongside its native
/completion which this project does not use -- see
docs/engine-performance-plan.md's '比較可能性の規律'), so one HTTP adapter
covers both engines; only base_url/model/name differ per instance.

Per the ModelAdapter contract (adapters/base.py), generate() hands back the
raw `message` object the server returned, JSON-serialized, unparsed --
OpenAIToolCallDecoder (decoders/base.py) is what turns it into a Prediction.
"""
from __future__ import annotations
import json
from typing import Any

from eval_bench.adapters.base import ModelAdapter


class OpenAICompatibleChatAdapter(ModelAdapter):
    name = "openai-compatible-chat"

    def __init__(self, base_url: str, model: str, sampling_params: dict[str, Any],
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sampling_params = sampling_params
        self.timeout = timeout

    def generate(self, query: str, tool_set: list[dict[str, Any]]) -> str:
        import requests  # lazy: decoder-only / transformers_hf-only users don't need this

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": query}],
            "tools": tool_set,
            "temperature": self.sampling_params["temperature"],
            "top_p": self.sampling_params["top_p"],
            "max_tokens": self.sampling_params["max_tokens"],
        }
        if self.sampling_params.get("seed") is not None:
            payload["seed"] = self.sampling_params["seed"]

        response = requests.post(
            f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        return json.dumps(message)


class VLLMAdapter(OpenAICompatibleChatAdapter):
    name = "vllm"


class LlamaCppAdapter(OpenAICompatibleChatAdapter):
    name = "llama_cpp"
