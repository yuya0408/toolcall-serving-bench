"""eval_bench/capacity.py: the capacity pre-check and post-check (docs/adr/0005).

The /metrics and startup-log samples are in vLLM 0.31.0's format as seen in
the 2026-10-18 smoke run (runs/gpu-20261018/)."""
from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

import pytest

from eval_bench import capacity
from eval_bench.adapters.openai_compatible import FINISH_REASON_KEY, USAGE_KEY, VLLMAdapter
from eval_bench.dataset.loader import load_cases

_METRICS = """\
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",enable_prefix_caching="True",engine="0",gpu_memory_utilization="0.92",num_cpu_blocks="None",num_gpu_blocks="6222"} 1.0
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 4.0
vllm:request_queue_time_seconds_sum{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 12.5
vllm:request_queue_time_seconds_count{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 400.0
vllm:request_queue_time_seconds_bucket{engine="0",le="0.3",model_name="Qwen/Qwen3-8B-AWQ"} 390.0
vllm:request_queue_time_seconds_bucket{engine="0",le="+Inf",model_name="Qwen/Qwen3-8B-AWQ"} 400.0
"""


def test_vllm_capacity_reads_blocks_from_cache_config_info():
    assert capacity.vllm_capacity(_METRICS) == {"unit": "block", "unit_tokens": 16, "units": 6222,
                                                "tokens": 99552}


def test_vllm_capacity_refuses_metrics_without_the_block_count():
    with pytest.raises(ValueError, match="num_gpu_blocks"):
        capacity.vllm_capacity('vllm:cache_config_info{block_size="16",num_gpu_blocks="None"} 1.0\n')


def test_vllm_counter_delta_over_a_cell():
    before = capacity.vllm_counters(_METRICS)
    after = capacity.vllm_counters(_METRICS.replace("} 4.0", "} 6.0").replace("} 400.0", "} 600.0"))
    delta = capacity.vllm_counter_delta(before, after)
    assert delta["preemptions"] == 2
    assert delta["queue_time"]["count"] == 200 and delta["queue_time"]["buckets"]["+Inf"] == 200


def test_startup_log_figure_tolerates_commas_and_misses_quietly():
    log = ("INFO [kv_cache_utils.py:2464] GPU KV cache size: 99,552 tokens, "
           "Maximum concurrency for 8,192 tokens per request: 12.15x\n")
    assert capacity.startup_log_max_concurrency(log) == 12.15
    assert capacity.startup_log_max_concurrency("nothing here") is None


def test_required_tokens_round_each_request_up_to_whole_blocks():
    # ADR-0005 Consequences: 2,591 + 512 = 3,103; + one 16-token block, rounded up -> 3,120.
    per_request = capacity.per_request_max(2591, 512)
    assert per_request == 3103
    assert capacity.required_tokens(per_request, 1, 16) == 3120
    assert capacity.required_tokens(per_request, 32, 16) == 99840          # > 99,552: AWQ c32 is unsupported
    assert capacity.required_tokens(per_request, 9, 16) <= 30240 < capacity.required_tokens(per_request, 10, 16)


def test_llama_ctx_size_is_the_requirement_rounded_to_256():
    assert capacity.llama_ctx_size(3103, 16) == 49664      # 16 x 3,104 = 49,664
    assert capacity.llama_ctx_size(3103, 1) == 3328


def test_llama_capacity_is_the_ctx_size_and_props_is_the_per_slot_limit():
    # The 2026-10-21 apply: --ctx-size 49664 --parallel 16 --kv-unified, and /props
    # n_ctx capped at Qwen3-8B's training context.
    props = {"default_generation_settings": {"n_ctx": 40960}, "total_slots": 16}
    assert capacity.llama_capacity(props, 49664) == {"unit": "token", "unit_tokens": 1, "units": 49664,
                                                     "tokens": 49664, "per_slot_tokens": 40960, "slots": 16}


def test_llama_pre_check_fits_when_the_total_and_one_slot_both_hold():
    props = {"default_generation_settings": {"n_ctx": 40960}, "total_slots": 16}
    assert capacity.pre_check("llama_cpp", capacity.llama_capacity(props, 49664), 2591, 512, 16)["fits"]
    assert not capacity.pre_check("llama_cpp", capacity.llama_capacity(props, 49408), 2591, 512, 16)["fits"]
    short_slot = {"default_generation_settings": {"n_ctx": 3103}, "total_slots": 16}
    assert not capacity.pre_check("llama_cpp", capacity.llama_capacity(short_slot, 49664), 2591, 512, 16)["fits"]


def test_pre_check_fits_only_when_every_request_can_grow_to_the_longest():
    cap = capacity.vllm_capacity(_METRICS)
    assert capacity.pre_check("vllm", cap, 2591, 512, 31)["fits"]
    assert not capacity.pre_check("vllm", cap, 2591, 512, 32)["fits"]


def test_post_check_flags_preemptions_and_a_longer_prompt_than_planned():
    delta = {"preemptions": 0, "queue_time": {"sum_s": 0.0, "count": 0, "buckets": {}}}
    assert capacity.post_check("vllm", 2591, [2500, 2590], vllm_delta=delta)["ok"]
    bad = capacity.post_check("vllm", 2591, [2600], vllm_delta={**delta, "preemptions": 1})
    assert not bad["ok"] and len(bad["problems"]) == 2


def test_post_check_for_llama_cpp_counts_length_limited_responses():
    assert capacity.post_check("llama_cpp", 2591, [2587], length_finished=0)["ok"]
    assert not capacity.post_check("llama_cpp", 2591, [2587], length_finished=2)["ok"]


def test_the_committed_budget_matches_the_current_cases_and_catalog():
    # Editing a case or the catalog without recounting must fail here, not on the GPU.
    version = json.loads(open("schema/tools.json", encoding="utf-8").read())["version"]
    entry = capacity.load_prompt_budget(10, version, load_cases("data"))
    assert entry["max_prompt_tokens"] >= 2500


def test_a_stale_budget_is_refused(tmp_path):
    cases = load_cases("data")
    path = tmp_path / "prompt_budget.json"
    path.write_text(json.dumps({"10": {"max_prompt_tokens": 2591, "tool_catalog_version": "2.1.1",
                                       "cases_sha256": capacity.cases_digest(cases[1:])}}))
    with pytest.raises(ValueError, match="recount"):
        capacity.load_prompt_budget(10, "2.1.1", cases, path)


def test_accuracy_pass_raw_output_keeps_usage_and_finish_reason():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "tool_calls": []}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 2200, "completion_tokens": 512},
    }
    sampling = {"temperature": 0, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "max_tokens": 512, "seed": 0}
    with patch("requests.post", return_value=response):
        raw = json.loads(VLLMAdapter("http://x", "m", sampling).generate("q", []))
    assert raw[USAGE_KEY]["prompt_tokens"] == 2200 and raw[FINISH_REASON_KEY] == "length"
