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
            "warmup_excluded": True,
            "n_requests": 100,
            "ttft_ms": {"p50": 10, "p95": 20, "p99": 30},
            "total_latency_ms": {"p50": 100, "p95": 200, "p99": 300},
            "tokens_per_sec": {"p50": 50, "p95": 60, "p99": 70},
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
        "results": {
            "warmup_excluded": True,
            "n_requests": 100,
            "ttft_ms": {"p50": 10, "p95": 20, "p99": 30},
            "total_latency_ms": {"p50": 100, "p95": 200, "p99": 300},
            "tokens_per_sec": {"p50": 50, "p95": 60, "p99": 70},
        },
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
