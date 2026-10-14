"""End-to-end wiring test for scripts/run_cell.py with no GPU / engine.

The backend is swapped for an adapter that answers every case correctly and
nvidia-smi for a fixed answer; everything between (argument handling, catalog
choice, the runner's guards, raw-log trial recovery, manifest validation) is
the real code path the GPU run will take.
"""
from __future__ import annotations
import importlib.util
import json
import pathlib

import pytest

from eval_bench.adapters.base import ModelAdapter
from eval_bench.dataset.loader import load_cases
from eval_bench.decoders.base import IdentityDecoder

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_cell", _ROOT / "scripts" / "run_cell.py")
run_cell = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_cell)

_CASES = load_cases(_ROOT / "data")


class _OracleAdapter(ModelAdapter):
    """Answers each query with its case's expected call (IdentityDecoder shape)."""
    name = "oracle"

    def __init__(self):
        self._by_query = {c["query"]: c for c in _CASES}

    def generate(self, query, tool_set):
        case = self._by_query[query]
        args = {**case["expected_args"]["exact"], **case["expected_args"]["semantic"]}
        return json.dumps({"tool": case["expected_tool"], "args": args}, ensure_ascii=False)


@pytest.fixture
def fake_gpu_backend(monkeypatch):
    monkeypatch.setattr(run_cell, "_build_backend", lambda args, sampling: (_OracleAdapter(), IdentityDecoder()))
    monkeypatch.setattr(run_cell, "_query_gpu", lambda: ("NVIDIA L4", 22.5))


def _run(tmp_path, *extra):
    argv = ["--pass-type", "accuracy", "--engine", "vllm", "--base-url", "http://x",
            "--model", "Qwen/Qwen3-8B", "--quantization", "bf16-none", "--engine-version", "0.0.0",
            "--tool-stage", "10", "--n", "3", "--out-dir", str(tmp_path), *extra]
    assert run_cell.main(argv) == 0
    manifest_path, = tmp_path.glob("*.json")
    raw_path, = tmp_path.glob("*.raw.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    return manifest, rows


def test_run_cell_writes_a_valid_manifest_and_a_complete_raw_log(tmp_path, fake_gpu_backend):
    manifest, rows = _run(tmp_path, "--run-id-prefix", "smoke-")

    assert manifest["run_id"].startswith("smoke-accuracy-vllm-")
    assert manifest["cell"]["tool_catalog_version"] == "2.1.0"   # stage 10 -> original catalog
    assert "concurrency" not in manifest["cell"]                  # forbidden on plain accuracy
    assert manifest["cell"]["sampling_params"]["top_k"] == 20
    assert manifest["repetition"] == {"n": 3, "seed_base": 0}
    assert manifest["hardware"]["gpu"] == "NVIDIA L4"
    assert manifest["results"]["selection_mean"] == 1.0

    # One row per (case, trial), every trial index recovered from the tool order.
    assert len(rows) == len(_CASES) * 3
    for case in _CASES:
        trials = sorted(r["trial"] for r in rows if r["case_id"] == case["id"])
        assert trials == [0, 1, 2]


def test_seed_base_changes_the_logged_orders(tmp_path, fake_gpu_backend):
    _, rows_a = _run(tmp_path / "a")
    _, rows_b = _run(tmp_path / "b", "--seed-base", "1")
    orders_a = {(r["case_id"], r["trial"]): r["tool_order"] for r in rows_a}
    orders_b = {(r["case_id"], r["trial"]): r["tool_order"] for r in rows_b}
    assert orders_a != orders_b
    assert all(sorted(orders_a[k]) == sorted(orders_b[k]) for k in orders_a)


@pytest.mark.parametrize("bad", [
    ["--pass-type", "accuracy", "--concurrency", "4"],                      # accuracy is serial
    ["--pass-type", "accuracy_under_load", "--concurrency", "16"],           # vllm >1 needs the log
    ["--engine", "llama_cpp"],                                               # needs --base-url
])
def test_argument_guards(bad):
    base = {"--pass-type": "accuracy", "--engine": "vllm", "--base-url": "http://x",
            "--model": "m", "--quantization": "q", "--engine-version": "v", "--tool-stage": "10"}
    pairs = dict(zip(bad[::2], bad[1::2]))
    if pairs.get("--engine") == "llama_cpp":
        base.pop("--base-url")
    merged = {**base, **pairs}
    argv = [tok for kv in merged.items() for tok in kv]
    with pytest.raises(SystemExit):
        run_cell._parse_args(argv)
