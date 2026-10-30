"""Structural guards on schema/run_manifest.schema.json's allOf rules. These
exist because a concurrency-under-load data point that got silently pooled
with the plain accuracy sweep's implicit concurrency=1 baseline would corrupt
the engine-performance-plan comparison without raising any error -- see
the engine-performance plan (kept outside this repo)'s 'KVキャッシュブロッカーの解消' and the
schema's own allOf $comments. Each guard is exercised on both a passing and a
failing manifest so a future edit that quietly weakens one of them fails loud.
"""
import copy
import json
import pathlib

import jsonschema
import pytest

from eval_bench.latency import TimedResult, summarize

_SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "schema" / "run_manifest.schema.json"
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

_HARDWARE = {"instance_type": "g6.xlarge", "gpu": "L4", "vram_gb": 22, "region": "ap-northeast-1"}
_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0, "max_tokens": 256, "seed": 0}
_ACCURACY_RESULTS = {
    "stage_size": 10,
    "n_cases": 20,
    "n_trials_per_case": 5,
    "selection_mean": 0.9,
    "selection_ci95": [0.85, 0.95],
    "arg_exact_mean": 0.8,
    "arg_semantic_mean": None,
    "consistency": 0.95,
}


def _validate(manifest):
    jsonschema.validate(instance=manifest, schema=_SCHEMA)


def _base_accuracy_manifest():
    return {
        "run_id": "run-1",
        "created_at": "2026-09-10T00:00:00Z",
        "pass_type": "accuracy",
        "cell": {
            "engine": "vllm",
            "engine_version": "abc123",
            "model": "Qwen/Qwen3-8B",
            "quantization": "awq-4bit",
            "sampling_params": _SAMPLING,
            "prompt_layout": "stable_first",
            "tool_stage": 10,
            "tool_catalog_version": "1.0.0",
        },
        "hardware": _HARDWARE,
        "repetition": {"n": 5, "seed_base": 0},
        "results": _ACCURACY_RESULTS,
    }


def test_plain_accuracy_manifest_is_valid():
    _validate(_base_accuracy_manifest())


def test_plain_accuracy_manifest_rejects_concurrency():
    manifest = _base_accuracy_manifest()
    manifest["cell"]["concurrency"] = 1
    with pytest.raises(jsonschema.ValidationError):
        _validate(manifest)


def _latency_results(cached: bool = True) -> dict:
    """What eval_bench.latency.summarize actually emits, so this schema test
    breaks if the code and the schema drift apart."""
    measured = [TimedResult(status="ok", total_s=1.0 + i / 1000, ttft_s=0.2, output_tokens=40,
                            prompt_tokens=2200, cached_tokens=2000 if cached else None)
                for i in range(299)]
    measured[0] = TimedResult(status="timeout", total_s=120.0)
    warmup = {"window_requests": 14, "warmup_requests": 14}
    results = summarize(measured, warmup)
    # run_latency_cell adds this after summarize.
    results["measured_stability"] = {"window_requests": 14, "stability_pct": 10.0, "stable": True,
                                     "windows": [{"throughput_rps": 1.0, "mean_latency_s": 1.0}] * 21}
    return results


def test_latency_results_without_prefix_cache_are_valid():
    manifest = {
        "run_id": "run-2b",
        "created_at": "2026-10-15T00:00:00Z",
        "pass_type": "latency",
        "cell": {
            "engine": "transformers_hf",
            "engine_version": "4.56.0",
            "model": "Qwen/Qwen3-8B",
            "quantization": "bf16-none",
            "sampling_params": _SAMPLING,
            "prompt_layout": "stable_first",
            "tool_stage": 10,
            "tool_catalog_version": "2.1.0",
            "concurrency": 1,
        },
        "hardware": _HARDWARE,
        "results": _latency_results(cached=False),
    }
    assert "prefix_cache" not in manifest["results"]
    _validate(manifest)


def _capacity_check():
    return {
    "engine": "vllm",
    "capacity": {"unit": "block", "unit_tokens": 16, "units": 6222, "tokens": 99552},
    "required": {"concurrency": 16, "per_request_tokens": 3103, "margin_units": 1, "tokens": 49920},
    "max_prompt_tokens_planned": 2591,
    "fits": True,
    "restarts": 0,
    "startup_log_max_concurrency": 12.15,
    "post_check": {"ok": True, "problems": [], "max_prompt_tokens_observed": 2590, "preemptions": 0,
                   "queue_time": {"sum_s": 1.5, "count": 2000, "buckets": {"0.3": 1990, "+Inf": 2000}}},
}


def test_latency_manifest_requires_concurrency_and_accepts_capacity_check():
    manifest = {
        "run_id": "run-2",
        "created_at": "2026-09-10T00:00:00Z",
        "pass_type": "latency",
        "cell": {
            "engine": "vllm",
            "engine_version": "abc123",
            "model": "Qwen/Qwen3-8B",
            "quantization": "awq-4bit",
            "sampling_params": _SAMPLING,
            "prompt_layout": "stable_first",
            "tool_stage": 10,
            "tool_catalog_version": "1.0.0",
            "concurrency": 4,
        },
        "hardware": _HARDWARE,
        "results": {
            **_latency_results(),
            "capacity_check": {**_capacity_check(), "required": {**_capacity_check()["required"], "concurrency": 4}},
            "reference_accuracy": {"n": 299, "selection_mean": 0.9, "arg_exact_mean": 0.8},
        },
    }
    _validate(manifest)


def test_capacity_check_accepts_tensorrt_llm_active_requests():
    check = {**_capacity_check(), "engine": "tensorrt_llm",
             "post_check": {"ok": True, "problems": [], "max_prompt_tokens_observed": 2500,
                            "active_requests": {"iterations": 307, "max_active": 16, "max_queued": 15}}}
    manifest = {
        "run_id": "run-trt",
        "created_at": "2026-10-28T00:00:00Z",
        "pass_type": "latency",
        "cell": {
            "engine": "tensorrt_llm",
            "engine_version": "1.2.1",
            "model": "Qwen/Qwen3-8B",
            "quantization": "bf16-none",
            "sampling_params": _SAMPLING,
            "prompt_layout": "stable_first",
            "tool_stage": 10,
            "tool_catalog_version": "1.0.0",
            "concurrency": 16,
        },
        "hardware": _HARDWARE,
        "results": {**_latency_results(), "capacity_check": check,
                    "reference_accuracy": {"n": 299, "selection_mean": 0.9, "arg_exact_mean": 0.8}},
    }
    _validate(manifest)


def test_latency_manifest_without_concurrency_is_invalid():
    manifest = {
        "run_id": "run-3",
        "created_at": "2026-09-10T00:00:00Z",
        "pass_type": "latency",
        "cell": {
            "engine": "vllm",
            "engine_version": "abc123",
            "model": "Qwen/Qwen3-8B",
            "quantization": "awq-4bit",
            "sampling_params": _SAMPLING,
            "prompt_layout": "stable_first",
            "tool_stage": 10,
            "tool_catalog_version": "1.0.0",
        },
        "hardware": _HARDWARE,
        "results": _latency_results(),
    }
    with pytest.raises(jsonschema.ValidationError):
        _validate(manifest)


def _base_accuracy_under_load_manifest():
    return {
        "run_id": "run-4",
        "created_at": "2026-09-10T00:00:00Z",
        "pass_type": "accuracy_under_load",
        "cell": {
            "engine": "vllm",
            "engine_version": "abc123",
            "model": "Qwen/Qwen3-8B",
            "quantization": "awq-4bit",
            "sampling_params": _SAMPLING,
            "prompt_layout": "stable_first",
            "tool_stage": 100,
            "tool_catalog_version": "1.0.0",
            "concurrency": 16,
        },
        "hardware": _HARDWARE,
        "repetition": {"n": 5, "seed_base": 0},
        "results": {
            **_ACCURACY_RESULTS,
            "capacity_check": _capacity_check(),
        },
    }


def test_accuracy_under_load_manifest_is_valid():
    _validate(_base_accuracy_under_load_manifest())


def test_accuracy_under_load_accepts_llama_cpp():
    # ADR-0002 addendum 2026-10-19: every configuration that holds concurrency 16.
    manifest = _base_accuracy_under_load_manifest()
    manifest["cell"]["engine"] = "llama_cpp"
    manifest["cell"]["quantization"] = "gguf-bf16"
    manifest["results"]["capacity_check"] = {
        **_capacity_check(), "engine": "llama_cpp",
        "capacity": {"unit": "token", "unit_tokens": 1, "units": 49920, "tokens": 49920, "slots": 16},
        "post_check": {"ok": True, "problems": [], "max_prompt_tokens_observed": 2586, "length_finished": 0},
    }
    _validate(manifest)


def test_accuracy_under_load_rejects_transformers_hf():
    manifest = _base_accuracy_under_load_manifest()
    manifest["cell"]["engine"] = "transformers_hf"
    with pytest.raises(jsonschema.ValidationError):
        _validate(manifest)


def test_accuracy_under_load_requires_concurrency():
    manifest = _base_accuracy_under_load_manifest()
    del manifest["cell"]["concurrency"]
    with pytest.raises(jsonschema.ValidationError):
        _validate(manifest)


def test_capacity_check_rejects_missing_post_check():
    manifest = _base_accuracy_under_load_manifest()
    del manifest["results"]["capacity_check"]["post_check"]
    with pytest.raises(jsonschema.ValidationError):
        _validate(manifest)


def test_capacity_check_rejects_a_failed_pre_check():
    manifest = _base_accuracy_under_load_manifest()
    manifest["results"]["capacity_check"]["fits"] = False
    with pytest.raises(jsonschema.ValidationError):
        _validate(manifest)


if __name__ == "__main__":
    test_plain_accuracy_manifest_is_valid()
    test_plain_accuracy_manifest_rejects_concurrency()
    test_latency_manifest_requires_concurrency_and_accepts_capacity_check()
    test_latency_manifest_without_concurrency_is_invalid()
    test_accuracy_under_load_manifest_is_valid()
    test_accuracy_under_load_accepts_llama_cpp()
    test_accuracy_under_load_rejects_transformers_hf()
    test_accuracy_under_load_requires_concurrency()
    test_capacity_check_rejects_missing_post_check()
    print("run_manifest schema guards ok")
