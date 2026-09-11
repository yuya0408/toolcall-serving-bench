"""Tests for eval_bench/runner.py: the end-to-end orchestration from cases
to a schema-valid manifest, and the three invariant guards nothing enforced
before this (docs/engine-performance-plan.md's stage=10 catalog rule, the
accuracy_under_load tool_stage=10 pin, and llama_cpp's commit-hash
engine_version). EchoAdapter/a stub predict function exercise this without
any GPU or live engine -- see runner.py's module docstring.
"""
from __future__ import annotations
import json

import jsonschema
import pytest

from eval_bench.adapters.base import EchoAdapter
from eval_bench.dataset.loader import load_cases
from eval_bench.dataset.tools import load_tools
from eval_bench.decoders.base import IdentityDecoder
from eval_bench.pipeline import make_predict_fn
from eval_bench.runner import build_manifest, run_accuracy_cell, write_manifest
from eval_bench.scorers.separated import Prediction

_HARDWARE = {"instance_type": "g6.xlarge", "gpu": "L4", "vram_gb": 22, "region": "ap-northeast-1"}
_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256, "seed": 0}


def _judge(exp, pred):
    return exp.strip() in pred or pred.strip() in exp


def _base_cell(**overrides):
    cell = {
        "engine": "vllm",
        "engine_version": "abc1234",
        "model": "Qwen/Qwen3-8B",
        "quantization": "awq-4bit",
        "sampling_params": _SAMPLING,
        "prompt_layout": "stable_first",
        "tool_stage": 10,
        "tool_catalog_version": "1.0.0",
    }
    cell.update(overrides)
    return cell


def _always_correct_predict(case, tool_names):
    if case["expected_tool"] is None:
        return Prediction(tool=None, args={})
    exact = case["expected_args"].get("exact", {})
    semantic = case["expected_args"].get("semantic", {})
    return Prediction(tool=case["expected_tool"], args={**exact, **semantic})


def test_run_accuracy_cell_end_to_end_produces_aggregate_shape():
    cases = load_cases("data")
    tools = load_tools()
    results = run_accuracy_cell(
        cases, tools, _base_cell(), pass_type="accuracy",
        predict=_always_correct_predict, judge=_judge, n=5,
    )
    assert results["n_cases"] == len(cases)
    assert results["selection_mean"] == 1.0


def test_run_accuracy_cell_works_through_the_real_pipeline_glue_with_echo_adapter():
    """Confirms the whole chain -- adapter -> decoder -> pipeline.make_predict_fn
    -> runner -- is actually wired, not just individually unit-tested."""
    cases = load_cases("data")
    tools = load_tools()
    predict = make_predict_fn(EchoAdapter(), IdentityDecoder(), tools)
    results = run_accuracy_cell(
        cases, tools, _base_cell(), pass_type="accuracy",
        predict=predict, judge=_judge, n=2,
    )
    assert results["n_cases"] == len(cases)
    # EchoAdapter always returns "{}" -> Prediction(tool=None, ...). That's a
    # selection failure on every positive case, but a correct decline on every
    # negative case (expected_tool=None too) -- so the expected rate is exactly
    # the negative-case fraction of the dataset, not 0.0.
    n_negative = sum(1 for c in cases if c["negative"])
    assert results["selection_mean"] == pytest.approx(n_negative / len(cases))


def test_stage_10_rejects_the_extended_catalog():
    cases = load_cases("data")
    extended = load_tools("schema/tools_extended.json")
    with pytest.raises(ValueError, match="original 40-tool catalog"):
        run_accuracy_cell(
            cases, extended, _base_cell(tool_stage=10), pass_type="accuracy",
            predict=_always_correct_predict, judge=_judge, n=2,
        )


def test_stage_50_with_extended_catalog_is_fine():
    cases = load_cases("data")
    extended = load_tools("schema/tools_extended.json")
    results = run_accuracy_cell(
        cases, extended, _base_cell(tool_stage=50), pass_type="accuracy",
        predict=_always_correct_predict, judge=_judge, n=2,
    )
    assert results["n_cases"] == len(cases)


def test_accuracy_under_load_rejects_non_stage_10():
    cases = load_cases("data")
    extended = load_tools("schema/tools_extended.json")  # stage=50 needs the extended catalog
    with pytest.raises(ValueError, match="pinned to tool_stage=10"):
        run_accuracy_cell(
            cases, extended, _base_cell(tool_stage=50), pass_type="accuracy_under_load",
            predict=_always_correct_predict, judge=_judge, n=2,
        )


def test_accuracy_under_load_accepts_stage_10():
    cases = load_cases("data")
    tools = load_tools()
    results = run_accuracy_cell(
        cases, tools, _base_cell(tool_stage=10), pass_type="accuracy_under_load",
        predict=_always_correct_predict, judge=_judge, n=2,
    )
    assert results["n_cases"] == len(cases)


def test_llama_cpp_requires_commit_hash_engine_version():
    cases = load_cases("data")
    tools = load_tools()
    with pytest.raises(ValueError, match="commit hash"):
        run_accuracy_cell(
            cases, tools, _base_cell(engine="llama_cpp", engine_version="v1.2.3"),
            pass_type="accuracy", predict=_always_correct_predict, judge=_judge, n=2,
        )


def test_llama_cpp_accepts_a_real_commit_hash():
    cases = load_cases("data")
    tools = load_tools()
    results = run_accuracy_cell(
        cases, tools, _base_cell(engine="llama_cpp", engine_version="a3f9c21"),
        pass_type="accuracy", predict=_always_correct_predict, judge=_judge, n=2,
    )
    assert results["n_cases"] == len(cases)


def test_build_manifest_validates_against_schema():
    cases = load_cases("data")
    tools = load_tools()
    results = run_accuracy_cell(
        cases, tools, _base_cell(), pass_type="accuracy",
        predict=_always_correct_predict, judge=_judge, n=5,
    )
    manifest = build_manifest(
        run_id="run-test-1", pass_type="accuracy", cell=_base_cell(), hardware=_HARDWARE,
        results=results, repetition={"n": 5, "seed_base": 0},
    )
    assert manifest["run_id"] == "run-test-1"
    assert manifest["cell"]["tool_stage"] == 10


def test_build_manifest_rejects_a_malformed_manifest():
    with pytest.raises(jsonschema.ValidationError):
        build_manifest(
            run_id="run-test-2", pass_type="accuracy", cell=_base_cell(), hardware=_HARDWARE,
            results={"missing": "required fields"},
        )


def test_write_manifest_round_trips(tmp_path):
    manifest = {"run_id": "run-test-3", "created_at": "2026-09-11T00:00:00Z"}
    path = write_manifest(manifest, tmp_path)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
