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
import time
from typing import Any, Optional

from eval_bench.adapters.base import ModelAdapter
from eval_bench.latency import TimedResult

# Reserved keys generate() adds to the returned message JSON.
USAGE_KEY = "_usage"
FINISH_REASON_KEY = "_finish_reason"


class OpenAICompatibleChatAdapter(ModelAdapter):
    name = "openai-compatible-chat"

    def __init__(self, base_url: str, model: str, sampling_params: dict[str, Any],
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sampling_params = sampling_params
        self.timeout = timeout

    def build_payload(self, query: str, tool_set: list[dict[str, Any]]) -> dict[str, Any]:
        """The request body both generate() and generate_timed() send, so the
        accuracy and latency passes cannot drift apart in what they ask for."""
        # Every sampling knob is sent explicitly, never left to the server:
        # unset knobs fall back to per-engine defaults that differ (llama.cpp
        # server: top_k=40/min_p=0.05; vLLM: the model's generation_config,
        # top_k=20/min_p=0), which would make "engine delta" partly a
        # sampling-config delta. Same for thinking: Qwen3's chat template
        # defaults to thinking ON, and transformers_hf passes
        # enable_thinking=False, so this side must too
        # (tests/test_adapters.py's parity test pins all three engines).
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": query}],
            "tools": tool_set,
            "temperature": self.sampling_params["temperature"],
            "top_p": self.sampling_params["top_p"],
            "top_k": self.sampling_params["top_k"],
            "min_p": self.sampling_params["min_p"],
            "max_tokens": self.sampling_params["max_tokens"],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.sampling_params.get("seed") is not None:
            payload["seed"] = self.sampling_params["seed"]
        return payload

    def generate(self, query: str, tool_set: list[dict[str, Any]]) -> str:
        import requests  # lazy: decoder-only / transformers_hf-only users don't need this

        payload = self.build_payload(query, tool_set)
        response = requests.post(
            f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        message = dict(choice["message"])
        # Kept beside the message for the raw-output log (docs/adr/0005 3: the
        # observed prompt length and length-limited responses). The decoder
        # reads only tool_calls, so these keys never reach scoring.
        message[USAGE_KEY] = body.get("usage")
        message[FINISH_REASON_KEY] = choice.get("finish_reason")
        return json.dumps(message)


    def generate_timed(self, query: str, tool_set: list[dict[str, Any]],
                       timeout: float) -> TimedResult:
        """The latency pass's request: the same body as generate(), streamed,
        timed on arrival of each server-sent event.

        `timeout` bounds the whole request. Past it the request is a timeout
        (an exceedance, eval_bench/latency.py); any other failure raises, and
        the driver aborts the cell.
        """
        import requests

        payload = self.build_payload(query, tool_set)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        started = time.perf_counter()
        ttft = first_delta = last_choice_at = None
        usage: Optional[dict[str, Any]] = None
        timings: Optional[dict[str, Any]] = None
        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        finish_reason: Optional[str] = None

        def timed_out() -> TimedResult:
            return TimedResult(status="timeout", total_s=time.perf_counter() - started,
                               ttft_s=ttft, first_delta_s=first_delta)

        try:
            with requests.post(f"{self.base_url}/v1/chat/completions", json=payload,
                               stream=True, timeout=(10, timeout)) as response:
                response.raise_for_status()
                # chunk_size=None: hand over each chunk as it arrives. The
                # default (512 bytes) would hold early, short events back
                # until 512 bytes accumulate and inflate TTFT.
                for line in response.iter_lines(chunk_size=None):
                    now = time.perf_counter()
                    if now - started > timeout:
                        return timed_out()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[len(b"data:"):].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(f"engine error mid-stream: {event['error']}")
                    choices = event.get("choices") or []
                    if choices:
                        last_choice_at = now
                        if ttft is None:
                            ttft = now - started
                        delta = choices[0].get("delta") or {}
                        if delta.get("content"):
                            content.append(delta["content"])
                        if delta.get("reasoning_content"):
                            reasoning.append(delta["reasoning_content"])
                        if first_delta is None and (delta.get("content") or delta.get("tool_calls")
                                                    or delta.get("reasoning_content")):
                            first_delta = now - started
                        # Tool calls stream as fragments keyed by index: the name
                        # once, the arguments string in pieces (OpenAI's format).
                        for frag in delta.get("tool_calls") or []:
                            call = tool_calls.setdefault(frag.get("index", 0), {"name": "", "arguments": ""})
                            fn = frag.get("function") or {}
                            call["name"] += fn.get("name") or ""
                            call["arguments"] += fn.get("arguments") or ""
                        finish_reason = choices[0].get("finish_reason") or finish_reason
                    usage = event.get("usage") or usage
                    timings = event.get("timings") or timings
        except requests.exceptions.Timeout:
            return timed_out()
        except requests.exceptions.ConnectionError as exc:
            # A read timeout while streaming surfaces as ConnectionError
            # (requests wraps urllib3's ReadTimeoutError), not as Timeout.
            if "timed out" in str(exc).lower():
                return timed_out()
            raise
        # The request ends at its last choice-bearing event, as in vllm bench
        # serve: the trailing usage-only event carries no token.
        total = (last_choice_at or time.perf_counter()) - started
        if usage is None or "completion_tokens" not in usage:
            raise RuntimeError("the stream ended without a usage block (stream_options.include_usage)")
        return TimedResult(
            status="ok", total_s=total, ttft_s=ttft, first_delta_s=first_delta,
            output_tokens=usage["completion_tokens"], prompt_tokens=usage.get("prompt_tokens"),
            cached_tokens=self.cached_tokens(usage, timings),
            content="".join(content), reasoning="".join(reasoning),
            tool_calls=[tool_calls[i] for i in sorted(tool_calls)], finish_reason=finish_reason,
        )

    def cached_tokens(self, usage: dict[str, Any], timings: Optional[dict[str, Any]]) -> int:
        raise NotImplementedError


class VLLMAdapter(OpenAICompatibleChatAdapter):
    name = "vllm"

    def cached_tokens(self, usage, timings) -> int:
        details = usage.get("prompt_tokens_details") or {}
        if details.get("cached_tokens") is None:
            raise RuntimeError("vLLM reported no prompt_tokens_details.cached_tokens; "
                               "start the server with --enable-prompt-tokens-details")
        return details["cached_tokens"]


class LlamaCppAdapter(OpenAICompatibleChatAdapter):
    name = "llama_cpp"

    def cached_tokens(self, usage, timings) -> int:
        if not timings or timings.get("cache_n") is None:
            raise RuntimeError("llama-server reported no timings.cache_n")
        return timings["cache_n"]
