"""Structural guards on schema/run_manifest.schema.json's allOf rules. These
exist because a concurrency-under-load data point that got silently pooled
with the plain accuracy sweep's implicit concurrency=1 baseline would corrupt
the engine-performance-plan comparison without raising any error -- see
docs/engine-performance-plan.md's 'KVキャッシュブロッカーの解消' and the
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
    warmup = {"window_requests": 14, "windows": [{"throughput_rps": 1.0, "mean_latency_s": 1.0}] * 3,
              "warmup_requests": 42, "stability_pct": 10.0}
    return summarize(measured, warmup)


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


def test_latency_manifest_requires_concurrency_and_accepts_vllm_diagnostics():
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
            "vllm_diagnostics": {"gpu_blocks_reported": 1000, "observed_max_concurrency": 8},
        },
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
            "vllm_diagnostics": {"gpu_blocks_reported": 5000, "observed_max_concurrency": 20},
        },
    }


def test_accuracy_under_load_manifest_is_valid():
    _validate(_base_accuracy_under_load_manifest())


def test_accuracy_under_load_rejects_non_vllm_engine():
    manifest = _base_accuracy_under_load_manifest()
    manifest["cell"]["engine"] = "llama_cpp"
    with pytest.raises(jsonschema.ValidationError):
        _validate(manifest)


def test_accuracy_under_load_rejects_non_awq_quantization():
    manifest = _base_accuracy_under_load_manifest()
    manifest["cell"]["quantization"] = "fp16-none"
    with pytest.raises(jsonschema.ValidationError):
        _validate(manifest)


def test_accuracy_under_load_requires_concurrency():
    manifest = _base_accuracy_under_load_manifest()
    del manifest["cell"]["concurrency"]
    with pytest.raises(jsonschema.ValidationError):
        _validate(manifest)


def test_vllm_diagnostics_rejects_partial_fields():
    manifest = _base_accuracy_under_load_manifest()
    manifest["results"]["vllm_diagnostics"] = {"gpu_blocks_reported": 5000}
    with pytest.raises(jsonschema.ValidationError):
        _validate(manifest)


if __name__ == "__main__":
    test_plain_accuracy_manifest_is_valid()
    test_plain_accuracy_manifest_rejects_concurrency()
    test_latency_manifest_requires_concurrency_and_accepts_vllm_diagnostics()
    test_latency_manifest_without_concurrency_is_invalid()
    test_accuracy_under_load_manifest_is_valid()
    test_accuracy_under_load_rejects_non_vllm_engine()
    test_accuracy_under_load_rejects_non_awq_quantization()
    test_accuracy_under_load_requires_concurrency()
    test_vllm_diagnostics_rejects_partial_fields()
    print("run_manifest schema guards ok")
