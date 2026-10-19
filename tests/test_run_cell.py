"""End-to-end wiring test for scripts/run_cell.py with no GPU / engine.

The backend is swapped for an adapter that answers every case correctly and
nvidia-smi for a fixed answer; everything between (argument handling, catalog
choice, the runner's guards, raw-log trial recovery, manifest validation) is
the real code path the GPU run will take.
"""
from __future__ import annotations
import importlib.util
import itertools
import json
import pathlib

import pytest

from eval_bench import capacity
from eval_bench.adapters.base import ModelAdapter
from eval_bench.dataset.loader import load_cases
from eval_bench.decoders.base import IdentityDecoder, OpenAIToolCallDecoder
from eval_bench.latency import TimedResult

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
    assert manifest["cell"]["tool_catalog_version"] == "2.1.1"   # stage 10 -> original catalog
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


def test_max_cases_takes_the_first_cases_by_id(tmp_path, fake_gpu_backend):
    _, rows = _run(tmp_path, "--run-id-prefix", "smoke-", "--max-cases", "4")
    first = sorted(c["id"] for c in _CASES)[:4]
    assert sorted({r["case_id"] for r in rows}) == first
    assert len(rows) == 4 * 3


def test_max_cases_is_refused_on_a_main_run():
    argv = ["--pass-type", "accuracy", "--engine", "vllm", "--base-url", "http://x", "--model", "m",
            "--quantization", "q", "--engine-version", "v", "--tool-stage", "10", "--max-cases", "4"]
    with pytest.raises(SystemExit):
        run_cell._parse_args(argv)


@pytest.mark.parametrize("bad", [
    ["--pass-type", "accuracy", "--concurrency", "4"],                      # accuracy is serial
    ["--pass-type", "accuracy_under_load", "--concurrency", "16"],           # >1 needs --capacity-check
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


# ------------------------------------------------------------- latency pass

class _TimedOracle:
    """Answers instantly and correctly (streamed tool calls); the first request
    after the (faked) cache reset is cold."""

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []
        self.cases = {c["query"]: c for c in load_cases(_ROOT / "data")}

    def generate_timed(self, query, tool_set, timeout):
        self.calls.append((query, [t["function"]["name"] for t in tool_set]))
        cached = 0 if len(self.calls) == 1 else 2000
        case = self.cases[query.split("\n\n[request-id")[0]]
        calls = ([] if case["expected_tool"] is None else
                 [{"name": case["expected_tool"],
                   "arguments": json.dumps(case["expected_args"]["exact"], ensure_ascii=False)}])
        return TimedResult(status="ok", total_s=1.0, ttft_s=0.2, output_tokens=21,
                           prompt_tokens=2200, cached_tokens=cached, content="",
                           tool_calls=calls, finish_reason="tool_calls" if calls else "stop")


@pytest.fixture
def fake_latency_backend(monkeypatch):
    oracle = _TimedOracle()
    resets = []
    monkeypatch.setattr(run_cell, "_build_backend",
                        lambda args, sampling: (oracle, OpenAIToolCallDecoder()))
    monkeypatch.setattr(run_cell, "_query_gpu", lambda: ("NVIDIA L4", 22.5))
    monkeypatch.setattr(run_cell, "_reset_vllm_prefix_cache", resets.append)
    # The oracle answers instantly, so on the real clock each window's throughput is
    # microsecond jitter and the stability rule fails at random. A clock that moves
    # one second per reading makes every window identical.
    ticks = itertools.count()
    real = run_cell.run_latency_cell
    monkeypatch.setattr(run_cell, "run_latency_cell",
                        lambda *a, **k: real(*a, clock=lambda: float(next(ticks)), **k))
    return oracle, resets


def _run_latency(tmp_path, *extra):
    argv = ["--pass-type", "latency", "--engine", "vllm", "--base-url", "http://x",
            "--model", "Qwen/Qwen3-8B", "--quantization", "bf16-none", "--engine-version", "0.0.0",
            "--tool-stage", "10", "--n-requests", "28", "--run-id-prefix", "smoke-",
            "--out-dir", str(tmp_path), *extra]
    return run_cell.main(argv)


def test_latency_cell_resets_the_cache_and_writes_a_valid_manifest(tmp_path, fake_latency_backend):
    oracle, resets = fake_latency_backend
    assert _run_latency(tmp_path) == 0
    assert resets == ["http://x"]

    manifest = json.loads(next(tmp_path.glob("*Z.json")).read_text(encoding="utf-8"))
    assert manifest["pass_type"] == "latency" and manifest["cell"]["concurrency"] == 1
    assert manifest["results"]["n_requests"] == 28
    rows = [json.loads(line) for line in next(tmp_path.glob("*.raw.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert sum(r["phase"] == "measured" for r in rows) == 28
    # The measured requests' tool calls, scored as a reference (docs/adr/0003 J).
    assert manifest["results"]["reference_accuracy"] == {"n": 28, "selection_mean": 1.0, "arg_exact_mean": 1.0}
    assert "capacity_check" not in manifest["results"]   # concurrency 1

    # Same case -> same tools in the same order every time; the query is never repeated.
    by_case: dict[str, set] = {}
    for query, names in oracle.calls:
        by_case.setdefault(query.split("\n\n[request-id")[0], set()).add(tuple(names))
    assert all(len(orders) == 1 for orders in by_case.values())
    assert len({q for q, _ in oracle.calls}) == len(oracle.calls)


def test_aborted_latency_cell_is_recorded_as_unsupported(tmp_path, fake_latency_backend, monkeypatch):
    oracle, _ = fake_latency_backend

    def fail(query, tool_set, timeout):
        raise RuntimeError("HTTP 400: context length exceeded")
    monkeypatch.setattr(oracle, "generate_timed", fail)
    assert _run_latency(tmp_path) == 2
    aborted = json.loads(next(tmp_path.glob("*.aborted.json")).read_text(encoding="utf-8"))
    assert aborted["outcome"] == "unsupported" and "context length" in aborted["reason"]
    assert not list(tmp_path.glob("*Z.json"))


@pytest.mark.parametrize("bad", [
    ["--n-requests", "100", "--run-id-prefix", ""],                    # main runs need >= 299
    ["--engine", "transformers_hf", "--concurrency", "2"],             # control point is c=1 only
    [],                                                                # --n-requests is required
])
def test_latency_argument_guards(bad):
    base = {"--pass-type": "latency", "--engine": "vllm", "--base-url": "http://x", "--model": "m",
            "--quantization": "q", "--engine-version": "v", "--tool-stage": "10"}
    if bad:
        base["--n-requests"] = "299"
    merged = {**base, **dict(zip(bad[::2], bad[1::2]))}
    argv = [tok for kv in merged.items() for tok in kv]
    with pytest.raises(SystemExit):
        run_cell._parse_args(argv)


# ------------------------------------------------------------- capacity post-check (docs/adr/0005 3)

def _metrics(preemptions: float, queue_count: float) -> str:
    return (
        '# TYPE vllm:num_preemptions_total counter\n'
        f'vllm:num_preemptions_total{{engine="0",model_name="m"}} {preemptions}\n'
        f'vllm:request_queue_time_seconds_sum{{engine="0",model_name="m"}} {queue_count * 0.01}\n'
        f'vllm:request_queue_time_seconds_count{{engine="0",model_name="m"}} {queue_count}\n'
        f'vllm:request_queue_time_seconds_bucket{{engine="0",le="0.3",model_name="m"}} {queue_count}\n'
        f'vllm:request_queue_time_seconds_bucket{{engine="0",le="+Inf",model_name="m"}} {queue_count}\n'
    )


def _pre_check(tmp_path, concurrency=2):
    pre = capacity.pre_check("vllm", {"unit": "block", "unit_tokens": 16, "units": 6222, "tokens": 99552},
                             2591, 512, concurrency)
    pre.update(restarts=0, startup_log_max_concurrency=None)
    path = tmp_path / "capacity_check.json"
    path.write_text(json.dumps(pre))
    return path


@pytest.mark.parametrize("preempted, code, ok", [(0, 0, True), (3, 3, False)])
def test_latency_cell_writes_the_capacity_post_check(tmp_path, fake_latency_backend, monkeypatch,
                                                      preempted, code, ok):
    reads = iter([_metrics(10, 100), _metrics(10 + preempted, 160)])
    monkeypatch.setattr(run_cell.capacity, "fetch_metrics", lambda base_url: next(reads))
    check = _pre_check(tmp_path)
    assert _run_latency(tmp_path, "--concurrency", "2", "--capacity-check", str(check)) == code
    manifest = json.loads(next(tmp_path.glob("*Z.json")).read_text(encoding="utf-8"))
    post = manifest["results"]["capacity_check"]["post_check"]
    assert post["ok"] is ok and post["preemptions"] == preempted
    assert post["queue_time"]["count"] == 60 and post["max_prompt_tokens_observed"] == 2200


def test_a_capacity_check_for_another_concurrency_is_refused(tmp_path, fake_latency_backend):
    check = _pre_check(tmp_path, concurrency=4)
    with pytest.raises(SystemExit):
        _run_latency(tmp_path, "--concurrency", "2", "--capacity-check", str(check))
