"""The latency pass (eval_bench/latency.py, docs/adr/0003) with no GPU.

The statistics are checked against numbers ADR-0003 states; the driver runs
against a fake engine on a virtual clock, so warmup windows, the measured
slice and every abort path are deterministic.
"""
from __future__ import annotations
import itertools
import json
import sys
import threading
import types
from unittest.mock import MagicMock, patch

import pytest
import requests

from eval_bench.adapters.openai_compatible import LlamaCppAdapter, VLLMAdapter
from eval_bench.adapters.transformers_hf import TransformersHFAdapter
from eval_bench.latency import (
    SLO_CLASSES, CellAborted, TimedResult, clopper_pearson_one_sided, exceeds, is_stable,
    new_run_tag, percentiles, run_latency_cell, tagged_query, verdict,
)

_CASES = [{"id": f"case_{i:02d}", "query": f"q{i}"} for i in range(14)]
_CONV = SLO_CLASSES["conversational"]


# ---------------------------------------------------------------- statistics

def test_verdict_bands_at_n299_match_adr_0003():
    # ADR-0003 Consequences: n=299 -> 0 exceedances "meets", 1-6 "indeterminate", 7+ "fails".
    assert verdict(0, 299)["verdict"] == "meets"
    assert {verdict(k, 299)["verdict"] for k in range(1, 7)} == {"indeterminate"}
    assert verdict(7, 299)["verdict"] == "fails"


def test_299_is_the_smallest_n_that_can_meet():
    assert verdict(0, 298)["verdict"] == "indeterminate"
    assert verdict(0, 299)["verdict"] == "meets"


def test_clopper_pearson_zero_successes_has_closed_form():
    lower, upper = clopper_pearson_one_sided(0, 299)
    assert lower == 0.0
    assert upper == pytest.approx(1 - 0.05 ** (1 / 299), abs=1e-9)


def test_clopper_pearson_bounds_bracket_the_rate():
    lower, upper = clopper_pearson_one_sided(10, 300)
    assert lower < 10 / 300 < upper
    assert clopper_pearson_one_sided(300, 300)[1] == 1.0


def test_percentiles_interpolate_like_numpy():
    assert percentiles([1, 2, 3, 4, 5]) == {"p50": 3, "p95": pytest.approx(4.8), "p99": pytest.approx(4.96)}


def test_exceedance_counts_timeouts_and_either_limit():
    ok = TimedResult(status="ok", total_s=1.0, ttft_s=0.5, output_tokens=11)       # TPOT 50ms
    assert not exceeds(ok, _CONV)
    assert exceeds(TimedResult(status="timeout", total_s=120.0), _CONV)
    assert exceeds(TimedResult(status="ok", total_s=3.0, ttft_s=2.5, output_tokens=11), _CONV)
    assert exceeds(TimedResult(status="ok", total_s=2.5, ttft_s=0.5, output_tokens=11), _CONV)  # TPOT 200ms
    # One output token: TPOT is undefined, so only TTFT decides.
    one = TimedResult(status="ok", total_s=0.6, ttft_s=0.5, output_tokens=1)
    assert one.tpot_s is None and not exceeds(one, _CONV)


def test_stability_needs_three_windows_within_ten_percent():
    w = lambda rps, lat: {"throughput_rps": rps, "mean_latency_s": lat}  # noqa: E731
    assert not is_stable([w(1, 1), w(1, 1)])
    assert is_stable([w(5, 9), w(1.0, 1.0), w(1.05, 1.0), w(1.1, 1.08)])
    assert not is_stable([w(1.0, 1.0), w(1.0, 1.0), w(1.11, 1.0)])
    assert not is_stable([w(1.0, 1.0), w(1.0, 1.0), w(1.0, 1.2)])


def test_tagged_queries_are_unique_and_keep_the_query_first():
    tagged = {tagged_query("在庫は?", "ab12", i) for i in range(1000)}
    assert len(tagged) == 1000
    assert all(t.startswith("在庫は?") for t in tagged)


def test_run_tags_are_eight_digits():
    # Digits only: every value then costs the same number of tokens, so two
    # cells' prompts do not differ by the tag (scripts/count_prompt_tokens.py).
    tags = [new_run_tag() for _ in range(200)]
    assert all(len(tag) == 8 and tag.isdigit() for tag in tags)
    assert len(set(tags)) > 1


# -------------------------------------------------------------------- driver

class _VirtualEngine:
    """Every request takes `duration(index)` seconds of virtual time."""

    def __init__(self, duration=lambda i: 1.0, cached=lambda i: 0 if i == 0 else 2000):
        self.t = 0.0
        self.lock = threading.Lock()
        self.duration = duration
        self.cached = cached

    def clock(self) -> float:
        with self.lock:
            return self.t

    def send(self, case, index):
        d = self.duration(index)
        with self.lock:
            self.t += d
        return TimedResult(status="ok", total_s=d, ttft_s=0.2, output_tokens=21,
                           prompt_tokens=2200, cached_tokens=self.cached(index))


def test_steady_engine_measures_exactly_n_after_three_windows():
    engine = _VirtualEngine()
    rows, results = run_latency_cell(_CASES, engine.send, concurrency=1, n_requests=30,
                                     clock=engine.clock)
    assert results["warmup"]["warmup_requests"] == 3 * 14     # stable on the first 3 windows
    measured = [r for r in rows if r["phase"] == "measured"]
    assert [r["index"] for r in measured] == list(range(42, 72))
    assert results["n_requests"] == 30
    assert results["slo"]["conversational"]["n"] == 30
    assert results["prefix_cache"]["hit_token_ratio"] == pytest.approx(2000 / 2200)


def test_measured_slice_cycles_through_every_case():
    engine = _VirtualEngine()
    rows, _ = run_latency_cell(_CASES, engine.send, concurrency=1, n_requests=28, clock=engine.clock)
    measured = [r["case_id"] for r in rows if r["phase"] == "measured"]
    assert sorted(measured) == sorted([c["id"] for c in _CASES] * 2)


def test_concurrent_cell_finishes_every_measured_request():
    engine = _VirtualEngine()
    rows, results = run_latency_cell(_CASES, engine.send, concurrency=4, n_requests=40,
                                     clock=engine.clock)
    measured = [r for r in rows if r["phase"] == "measured"]
    assert len(measured) == 40 and results["n_requests"] == 40
    start = measured[0]["index"]
    assert [r["index"] for r in measured] == list(range(start, start + 40))


def test_unstable_engine_aborts_after_max_windows_and_keeps_the_log():
    # Alternating 1s / 2s windows: throughput and latency never settle within 10%.
    engine = _VirtualEngine(duration=lambda i: 1.0 if (i // 14) % 2 == 0 else 2.0)
    with pytest.raises(CellAborted, match="not stable") as exc_info:
        run_latency_cell(_CASES, engine.send, concurrency=1, n_requests=30, clock=engine.clock)
    assert len(exc_info.value.rows) == 10 * 14
    assert {r["phase"] for r in exc_info.value.rows} == {"warmup"}


def test_engine_error_aborts_the_cell():
    engine = _VirtualEngine()

    def send(case, index):
        if index == 20:
            raise RuntimeError("HTTP 400: context length exceeded")
        return engine.send(case, index)
    with pytest.raises(CellAborted, match="context length") as exc_info:
        run_latency_cell(_CASES, send, concurrency=2, n_requests=30, clock=engine.clock)
    assert len(exc_info.value.rows) >= 20


def test_warm_cache_at_cell_start_aborts():
    engine = _VirtualEngine(cached=lambda i: 1500)
    with pytest.raises(CellAborted, match="empty prefix cache"):
        run_latency_cell(_CASES, engine.send, concurrency=1, n_requests=30, clock=engine.clock)


def test_engine_without_prefix_cache_skips_the_cold_check():
    engine = _VirtualEngine(cached=lambda i: None)
    _, results = run_latency_cell(_CASES, engine.send, concurrency=1, n_requests=30, clock=engine.clock)
    assert "prefix_cache" not in results


# -------------------------------------------------------- streaming adapters

_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "max_tokens": 256, "seed": 0}
_TOOLS = [{"type": "function", "function": {"name": "get_inventory", "parameters": {}}}]


def _sse(*events) -> list[bytes]:
    return [b"data: " + json.dumps(e).encode() for e in events] + [b"data: [DONE]"]


def _streaming_post(lines):
    response = MagicMock()
    response.__enter__.return_value = response
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = iter(lines)
    return patch("requests.post", return_value=response)


_TOOL_CALL_STREAM = [
    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
    {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"name": "get_inventory"}}]}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
]


def test_vllm_stream_reads_ttft_tokens_and_cached_tokens():
    usage = {"prompt_tokens": 2200, "completion_tokens": 30,
             "prompt_tokens_details": {"cached_tokens": 2048}}
    adapter = VLLMAdapter("http://x", "Qwen/Qwen3-8B", _SAMPLING)
    with _streaming_post(_sse(*_TOOL_CALL_STREAM, {"choices": [], "usage": usage})) as post:
        result = adapter.generate_timed("在庫", _TOOLS, timeout=60)

    payload = post.call_args[1]["json"]
    assert payload == {**adapter.build_payload("在庫", _TOOLS), "stream": True,
                       "stream_options": {"include_usage": True}}
    assert post.call_args[1]["stream"] is True
    assert result.status == "ok"
    assert result.ttft_s is not None and result.first_delta_s >= result.ttft_s
    assert (result.output_tokens, result.prompt_tokens, result.cached_tokens) == (30, 2200, 2048)


def test_streamed_tool_call_fragments_are_assembled():
    # docs/adr/0003 J: the latency pass keeps the tool call for the reference accuracy.
    stream = [
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"name": "get_inventory", "arguments": ""}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"item'}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '": "A1"}'}}]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 9,
                                  "prompt_tokens_details": {"cached_tokens": 0}}},
    ]
    with _streaming_post(_sse(*stream)):
        result = VLLMAdapter("http://x", "m", _SAMPLING).generate_timed("q", _TOOLS, timeout=60)
    assert result.tool_calls == [{"name": "get_inventory", "arguments": '{"item": "A1"}'}]
    assert result.finish_reason == "tool_calls"


def test_vllm_without_prompt_tokens_details_is_a_configuration_error():
    usage = {"prompt_tokens": 2200, "completion_tokens": 30}
    with _streaming_post(_sse(*_TOOL_CALL_STREAM, {"choices": [], "usage": usage})):
        with pytest.raises(RuntimeError, match="--enable-prompt-tokens-details"):
            VLLMAdapter("http://x", "m", _SAMPLING).generate_timed("q", _TOOLS, timeout=60)


def test_llama_cpp_stream_reads_cache_n_from_timings():
    final = {"choices": [], "usage": {"prompt_tokens": 2200, "completion_tokens": 25},
             "timings": {"cache_n": 2100, "prompt_n": 100}}
    with _streaming_post(_sse(*_TOOL_CALL_STREAM, final)):
        result = LlamaCppAdapter("http://x", "m", _SAMPLING).generate_timed("q", _TOOLS, timeout=60)
    assert result.cached_tokens == 2100


def test_read_timeout_while_streaming_is_a_timeout_not_an_error():
    def lines():
        yield b"data: " + json.dumps(_TOOL_CALL_STREAM[0]).encode()
        raise requests.exceptions.ConnectionError("HTTPConnectionPool(host='x'): Read timed out.")
    with _streaming_post(lines()):
        result = VLLMAdapter("http://x", "m", _SAMPLING).generate_timed("q", _TOOLS, timeout=60)
    assert result.status == "timeout" and result.ttft_s is not None


def test_refused_connection_raises():
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(requests.exceptions.ConnectionError):
            VLLMAdapter("http://x", "m", _SAMPLING).generate_timed("q", _TOOLS, timeout=60)


def test_error_event_mid_stream_raises():
    with _streaming_post(_sse({"error": {"message": "out of memory"}})):
        with pytest.raises(RuntimeError, match="out of memory"):
            VLLMAdapter("http://x", "m", _SAMPLING).generate_timed("q", _TOOLS, timeout=60)


def test_transformers_hf_times_each_generated_token():
    class _Ids(list):
        shape = (1, 3)
        device = "cpu"

        def to(self, device):
            return self

    class _Batch(dict):
        def to(self, device):
            return self

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "PROMPT"

        def __call__(self, prompt, return_tensors="pt"):
            return _Batch(input_ids=_Ids([1, 2, 3]))

        def decode(self, ids, skip_special_tokens=True):
            return '<tool_call>{"name": "get_inventory"}</tool_call>'

    class _Model:
        device = "cpu"

        def generate(self, stopping_criteria, **kwargs):
            for _ in range(5):                     # 5 new tokens, criteria called after each
                for criterion in stopping_criteria:
                    criterion(_Ids([0]), None)
            return [[1, 2, 3, 9, 9, 9, 9, 9]]

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False, synchronize=lambda: None),
        zeros=lambda *a, **k: [False], bool=bool, manual_seed=lambda s: None)
    fake_transformers = types.SimpleNamespace(StoppingCriteriaList=list)
    with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
        result = TransformersHFAdapter(_Tokenizer(), _Model(), _SAMPLING).generate_timed(
            "q", _TOOLS, timeout=60)
    assert result.status == "ok"
    assert result.output_tokens == 5 and result.prompt_tokens == 3
    assert result.cached_tokens is None
    assert 0 <= result.ttft_s <= result.total_s
