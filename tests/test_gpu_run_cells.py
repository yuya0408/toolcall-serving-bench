"""scripts/gpu/run_cells.py without a GPU: the cell lists are valid, and each
cell becomes the server and run_cell.py command it should."""
from __future__ import annotations
import importlib.util
import json
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_cells", _ROOT / "scripts" / "gpu" / "run_cells.py")
run_cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_cells)

_spec_rc = importlib.util.spec_from_file_location("run_cell", _ROOT / "scripts" / "run_cell.py")
run_cell = importlib.util.module_from_spec(_spec_rc)
_spec_rc.loader.exec_module(run_cell)

_REAL_SWAP_DEVICES = run_cells.swap_devices   # kept before the autouse fixture replaces it
_VERSIONS = {"vllm": "0.20.0", "transformers": "4.57.0", "llama_cpp_commit": "0123456789abcdef"}


def _lists():
    return sorted((_ROOT / "scripts" / "gpu" / "cells").glob("*.json"))


@pytest.mark.parametrize("path", _lists(), ids=lambda p: p.stem)
def test_cell_lists_are_valid_and_every_cell_parses_as_run_cell_arguments(path, tmp_path):
    cells = json.loads(path.read_text(encoding="utf-8"))
    run_cells.validate(cells)
    for cell in cells:
        check = tmp_path / "capacity_check.json" if run_cells.needs_capacity_check(cell) else None
        cmd = run_cells.run_cell_command(cell, run_cells.engine_version(cell["engine"], _VERSIONS),
                                         tmp_path, check)
        run_cell._parse_args(cmd[2:])   # the same guards run_cell.py applies on the instance


def test_vllm_server_has_the_flags_the_adrs_require():
    cmd = run_cells.server_command({"engine": "vllm", "quantization": "awq-4bit",
                                    "server": {"max_model_len": 8192}})
    assert cmd[1:3] == ["serve", "Qwen/Qwen3-8B-AWQ"]
    for flag in ("--enable-auto-tool-choice", "--enable-prompt-tokens-details"):
        assert flag in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"
    assert cmd[cmd.index("--max-model-len") + 1] == "8192"


def test_llama_server_is_sized_to_the_requirement():
    # ADR-0005 1 and its 2026-10-23 addendum: one slot per request, each with its own
    # cache of (longest prompt + max_tokens + 1-token margin) padded to 256; --fit off.
    cmd = run_cells.server_command({"engine": "llama_cpp", "quantization": "gguf-bf16", "concurrency": 16},
                                   max_prompt_tokens=2591)
    assert "--jinja" in cmd and "--no-kv-unified" in cmd and "--metrics" in cmd
    assert "--kv-unified" not in cmd
    assert cmd[cmd.index("-lv") + 1] == "4"   # the resolved settings (flash attention, batch) reach the log
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("--parallel") + 1] == "16"
    assert cmd[cmd.index("--ctx-size") + 1] == "53248"     # 16 x 3,328
    unified = run_cells.server_command({"engine": "llama_cpp", "quantization": "gguf-bf16", "concurrency": 16},
                                       max_prompt_tokens=2591, kv="unified")
    assert "--kv-unified" in unified and unified[unified.index("--ctx-size") + 1] == "49664"
    assert cmd[cmd.index("-m") + 1].endswith("Qwen3-8B-bf16.gguf")
    one = run_cells.server_command({"engine": "llama_cpp", "quantization": "gguf-bf16"}, max_prompt_tokens=2591)
    assert one[one.index("--parallel") + 1] == "1" and one[one.index("--ctx-size") + 1] == "3328"


def test_llama_server_serves_the_gguf_of_the_cell_quantization():
    # docs/adr/0008: Q4_K_M is a second GGUF of the same model
    q4 = run_cells.server_command({"engine": "llama_cpp", "quantization": "gguf-q4_k_m"}, max_prompt_tokens=2591)
    assert q4[q4.index("-m") + 1].endswith("Qwen3-8B-Q4_K_M.gguf")
    assert run_cells.MODELS["gguf-q4_k_m"] == "Qwen/Qwen3-8B"


def test_transformers_hf_needs_no_server_and_a_loaded_cell_gets_its_capacity_check(tmp_path):
    assert run_cells.server_command({"engine": "transformers_hf", "quantization": "bf16-none"}) is None
    cell = {"name": "x", "prefix": "pilot-", "pass_type": "accuracy_under_load", "engine": "vllm",
            "quantization": "awq-4bit", "tool_stage": 10, "concurrency": 16, "n": 5, "temperature": 0}
    cmd = run_cells.run_cell_command(cell, "0.20.0", tmp_path, tmp_path / "capacity_check.json")
    assert cmd[cmd.index("--capacity-check") + 1].endswith("capacity_check.json")
    assert cmd[cmd.index("--engine-version") + 1] == "0.20.0"
    assert cmd[cmd.index("--model") + 1] == "Qwen/Qwen3-8B-AWQ"


@pytest.mark.parametrize("bad", [
    {"engine": "llama_cpp", "quantization": "bf16-none"},     # llama_cpp runs the GGUF only
    {"engine": "vllm", "quantization": "gguf-bf16"},          # and only llama_cpp runs it
    {"engine": "vllm", "quantization": "gguf-q4_k_m"},        # either GGUF
    {"prefix": "dry-"},
    {"servr": {}},                                            # a typo must not be ignored
    {"n": 1},                                                 # consistency needs two trials
    {"engine": "llama_cpp", "quantization": "gguf-bf16",
     "server": {"ctx_size": 40960}},                          # llama.cpp is sized from the budget
])
def test_validate_refuses(bad):
    cell = {"name": "x", "prefix": "smoke-", "pass_type": "accuracy", "engine": "vllm",
            "quantization": "bf16-none", "tool_stage": 10, **bad}
    with pytest.raises(SystemExit):
        run_cells.validate([cell])


def _hf_cell(name, **extra):
    return {"name": name, "prefix": "smoke-", "pass_type": "accuracy", "engine": "transformers_hf",
            "quantization": "bf16-none", "tool_stage": 10, "n": 2, **extra}


def _fake_run(monkeypatch, code, gpu_freed=True):
    """transformers_hf needs no server; run_cell.py is replaced by a tiny Python program."""
    import sys
    monkeypatch.setattr(run_cells, "run_cell_command", lambda *a: [sys.executable, "-c", code])
    monkeypatch.setattr(run_cells, "_wait_gpu_free", lambda: gpu_freed)


def test_a_cell_past_its_time_limit_is_stopped_and_recorded(tmp_path, monkeypatch):
    _fake_run(monkeypatch, "import time; time.sleep(60)")
    status = run_cells.run_one(_hf_cell("slow", timeout_min=0.02), _VERSIONS, tmp_path)
    assert status["outcome"] == "timeout" and status["gpu_freed"] is True


@pytest.fixture(autouse=True)
def _no_swap(monkeypatch):
    # The CI runner has swap on; the guard is tested on its own below.
    monkeypatch.setattr(run_cells, "swap_devices", lambda: [])


def _main(tmp_path, cells):
    (tmp_path / "versions.json").write_text(json.dumps(_VERSIONS))
    path = tmp_path / "cells.json"
    path.write_text(json.dumps(cells))
    return run_cells.main([str(path), "--out", str(tmp_path)])


def test_the_list_stops_with_failed_when_the_gpu_is_not_freed(tmp_path, monkeypatch):
    _fake_run(monkeypatch, "pass", gpu_freed=False)
    assert _main(tmp_path, [_hf_cell("a"), _hf_cell("b")]) == 1
    lines = (tmp_path / "status.jsonl").read_text().splitlines()
    assert [json.loads(l)["name"] for l in lines] == ["a"]          # b never started
    assert "GPU memory still in use after a" in (tmp_path / "FAILED").read_text()
    assert not (tmp_path / "RUNNING").exists() and not (tmp_path / "DONE").exists()


def test_a_crash_of_the_runner_itself_leaves_failed_not_running(tmp_path, monkeypatch):
    def boom(*a):
        raise RuntimeError("boom")
    monkeypatch.setattr(run_cells, "run_one", boom)
    with pytest.raises(RuntimeError):
        _main(tmp_path, [_hf_cell("a")])
    assert "boom" in (tmp_path / "FAILED").read_text()
    assert not (tmp_path / "RUNNING").exists()


def test_a_finished_list_leaves_done_and_clears_an_old_failed(tmp_path, monkeypatch):
    (tmp_path / "FAILED").write_text("from an earlier list")
    _fake_run(monkeypatch, "pass")
    assert _main(tmp_path, [_hf_cell("a")]) == 0
    assert (tmp_path / "DONE").exists() and not (tmp_path / "FAILED").exists()
    status = json.loads((tmp_path / "status.jsonl").read_text())
    assert status["outcome"] == "ok"
    assert status["started_at"] <= status["run_started_at"] <= status["finished_at"]


def test_swap_devices_reads_proc_swaps(tmp_path):
    swaps = tmp_path / "swaps"
    swaps.write_text("Filename\tType\tSize\tUsed\tPriority\n"
                     "/swapfile-setup                         file\t\t16777212\t0\t\t-2\n")
    assert _REAL_SWAP_DEVICES(swaps) == ["/swapfile-setup"]
    swaps.write_text("Filename\tType\tSize\tUsed\tPriority\n")
    assert _REAL_SWAP_DEVICES(swaps) == []
    assert _REAL_SWAP_DEVICES(tmp_path / "absent") == []


def test_no_cell_starts_while_swap_is_on(tmp_path, monkeypatch):
    _fake_run(monkeypatch, "pass")
    monkeypatch.setattr(run_cells, "swap_devices", lambda: ["/swapfile-setup"])
    assert _main(tmp_path, [_hf_cell("a")]) == 1
    assert "swap is active" in (tmp_path / "FAILED").read_text()
    assert not (tmp_path / "status.jsonl").exists()


# ---------------------------------------------------------------- the capacity pre-check (ADR-0005 2)

_BUDGET = {"max_prompt_tokens": 2591}


class _FakeServer:
    def poll(self):
        return None


def _served(monkeypatch, tmp_path, capacities):
    """run_one with the server and run_cell.py faked: each startup reports the
    next capacity in `capacities`; returns the startups, the run_cell commands
    and the order of the engine-state snapshots around the run."""
    seq = iter(capacities)
    starts, runs = [], []
    snapshots = []

    def start(cell, cmd, log, status):
        log.write_text("Maximum concurrency for 8,192 tokens per request: 12.15x\n")
        starts.append(1)
        return _FakeServer(), True

    def fake_run_cell_command(cell, version, cell_dir, check):
        runs.append(check)
        snapshots.append("run")
        return [sys.executable, "-c", "pass"]

    monkeypatch.setattr(run_cells, "_start_server", start)
    monkeypatch.setattr(run_cells, "read_capacity", lambda engine, ctx_size=None, kv="split": next(seq))
    monkeypatch.setattr(run_cells, "_stop", lambda proc: None)
    monkeypatch.setattr(run_cells, "_wait_gpu_free", lambda: True)
    monkeypatch.setattr(run_cells, "run_cell_command", fake_run_cell_command)
    monkeypatch.setattr(run_cells, "snapshot_engine_state",
                        lambda engine, cell_dir, tag: snapshots.append(tag) or [])
    return starts, runs, snapshots


def _vllm_cap(tokens):
    return {"unit": "block", "unit_tokens": 16, "units": tokens // 16, "tokens": tokens}


def _loaded(engine="vllm", quantization="awq-4bit", concurrency=16):
    return {"name": "load", "prefix": "pilot-", "pass_type": "accuracy_under_load", "engine": engine,
            "quantization": quantization, "tool_stage": 10, "concurrency": concurrency, "n": 2, "temperature": 0}


def test_a_short_vllm_startup_is_restarted_once_and_then_runs(tmp_path, monkeypatch):
    # The 2026-10-18 smoke: the first startup came out exactly 1 GiB short.
    starts, runs, snapshots = _served(monkeypatch, tmp_path, [_vllm_cap(48000), _vllm_cap(48000), _vllm_cap(99552), _vllm_cap(99552)])
    status = run_cells.run_one(_loaded(), _VERSIONS, tmp_path, _BUDGET)
    assert status["outcome"] == "ok" and len(starts) == 2
    check = json.loads((tmp_path / "load" / "capacity_check.json").read_text())
    assert check["fits"] and check["restarts"] == 1 and check["startup_log_max_concurrency"] == 12.15
    assert check["required"]["tokens"] == 16 * 3120
    assert (tmp_path / "load" / "server-short-startup.log").exists()
    assert runs == [tmp_path / "load" / "capacity_check.json"]


def test_a_cell_that_still_does_not_fit_is_unsupported_without_running(tmp_path, monkeypatch):
    starts, runs, snapshots = _served(monkeypatch, tmp_path, [_vllm_cap(99552)] * 4)
    status = run_cells.run_one(_loaded(concurrency=32), _VERSIONS, tmp_path, _BUDGET)
    assert status["outcome"] == "unsupported" and len(starts) == 2 and runs == []
    check = json.loads((tmp_path / "load" / "capacity_check.json").read_text())
    assert not check["fits"] and check["required"]["tokens"] == 32 * 3120   # 99,840 > 99,552


def test_llama_cpp_is_never_restarted_for_capacity(tmp_path, monkeypatch):
    llama = {"unit": "token", "unit_tokens": 1, "units": 53248, "tokens": 53248, "per_slot_tokens": 3328,
             "slots": 16}
    starts, runs, snapshots = _served(monkeypatch, tmp_path, [llama])
    status = run_cells.run_one(_loaded("llama_cpp", "gguf-bf16"), _VERSIONS, tmp_path, _BUDGET)
    assert status["outcome"] == "ok" and len(starts) == 1
    assert "kv_capacity_tokens" not in status
    # The engine's state is saved on both sides of the run.
    assert snapshots == ["before", "run", "after"] and status["engine_state_failed"] == []


def test_a_cell_skipped_as_unsupported_saves_no_engine_state(tmp_path, monkeypatch):
    starts, runs, snapshots = _served(monkeypatch, tmp_path, [_vllm_cap(99552)] * 4)
    run_cells.run_one(_loaded(concurrency=32), _VERSIONS, tmp_path, _BUDGET)
    assert snapshots == []


def test_engine_state_snapshot_saves_the_reads_and_returns_the_failures(tmp_path, monkeypatch):
    import requests

    class Response:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    def get(url, timeout):
        if url.endswith("/slots"):
            raise requests.ConnectionError("refused")
        return Response(url.rsplit("/", 1)[1])

    monkeypatch.setattr(requests, "get", get)
    failed = run_cells.snapshot_engine_state("llama_cpp", tmp_path, "before")
    assert (tmp_path / "engine-metrics-before.txt").read_text() == "metrics"
    assert (tmp_path / "engine-props-before.json").read_text() == "props"
    assert len(failed) == 1 and "/slots" in failed[0]
    assert run_cells.snapshot_engine_state("vllm", tmp_path, "after") == []
    assert sorted(f.name for f in tmp_path.iterdir()) == [
        "engine-metrics-after.txt", "engine-metrics-before.txt", "engine-props-before.json"]
    # vLLM's resolved config (max_num_batched_tokens and the rest) comes from its dev-mode /server_info.
    vllm_dir = tmp_path / "vllm"
    vllm_dir.mkdir()
    assert run_cells.snapshot_engine_state("vllm", vllm_dir, "before") == []
    assert (vllm_dir / "engine-server_info-before.json").read_text() == "server_info?config_format=json"


def test_llama_cpp_capacity_is_read_with_the_ctx_size_it_was_started_with(tmp_path, monkeypatch):
    seen = []
    starts, runs, snapshots = _served(monkeypatch, tmp_path, [])
    props = {"default_generation_settings": {"n_ctx": 3328}, "total_slots": 16}

    def read(engine, ctx_size=None, kv="split"):
        seen.append((ctx_size, kv))
        return run_cells.capacity.llama_capacity(props, ctx_size, kv)

    monkeypatch.setattr(run_cells, "read_capacity", read)
    status = run_cells.run_one(_loaded("llama_cpp", "gguf-bf16"), _VERSIONS, tmp_path, _BUDGET)
    assert status["outcome"] == "ok" and seen == [(53248, "split")] and status["llama_kv"] == "split"
    check = json.loads((tmp_path / "load" / "capacity_check.json").read_text())
    assert check["llama_kv"] == "split" and check["capacity"]["tokens"] == 53248


def _failing_split(monkeypatch, tmp_path):
    """run_one where llama.cpp's split-KV startup fails and the unified one starts."""
    commands = []
    seen = []

    def start(cell, cmd, log, status):
        commands.append(cmd)
        log.write_text("out of memory\n" if "--no-kv-unified" in cmd else "ok\n")
        return _FakeServer(), "--kv-unified" in cmd

    def read(engine, ctx_size=None, kv="split"):
        seen.append((ctx_size, kv))
        return run_cells.capacity.llama_capacity(
            {"default_generation_settings": {"n_ctx": 40960}, "total_slots": 16}, ctx_size, kv)

    monkeypatch.setattr(run_cells, "_start_server", start)
    monkeypatch.setattr(run_cells, "read_capacity", read)
    monkeypatch.setattr(run_cells, "_stop", lambda proc: None)
    monkeypatch.setattr(run_cells, "_wait_gpu_free", lambda: True)
    monkeypatch.setattr(run_cells, "run_cell_command", lambda *a: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(run_cells, "snapshot_engine_state", lambda engine, cell_dir, tag: [])
    return commands, seen


def test_a_split_llama_cache_that_does_not_start_falls_back_to_unified(tmp_path, monkeypatch):
    commands, seen = _failing_split(monkeypatch, tmp_path)
    status = run_cells.run_one(_loaded("llama_cpp", "gguf-bf16"), _VERSIONS, tmp_path, _BUDGET)
    assert status["outcome"] == "ok" and status["llama_kv"] == "unified"
    assert ["--no-kv-unified" in c for c in commands] == [True, False]
    assert seen == [(49664, "unified")]
    cell_dir = tmp_path / "load"
    assert (cell_dir / "server-split-startup.log").read_text() == "out of memory\n"
    assert "--kv-unified" in (cell_dir / "server_command.txt").read_text()
    assert json.loads((cell_dir / "capacity_check.json").read_text())["llama_kv"] == "unified"


def test_a_single_slot_llama_cell_has_no_unified_fallback(tmp_path, monkeypatch):
    # With one slot the two layouts are the same cache: a failed start is just server_not_ready.
    commands, seen = _failing_split(monkeypatch, tmp_path)
    cell = {"name": "lat", "prefix": "pilot-", "pass_type": "latency", "engine": "llama_cpp",
            "quantization": "gguf-bf16", "tool_stage": 10, "concurrency": 1, "n_requests": 10}
    status = run_cells.run_one(cell, _VERSIONS, tmp_path, _BUDGET)
    assert status["outcome"] == "server_not_ready" and len(commands) == 1 and seen == []


def test_the_seed_check_starts_a_sized_llama_server(tmp_path, monkeypatch):
    # 2026-10-21: it called server_command without the budget and never ran.
    spec = importlib.util.spec_from_file_location("check_llama_seed",
                                                  _ROOT / "scripts" / "gpu" / "check_llama_seed.py")
    seed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed)
    commands = []

    def spawn(cmd, log):
        commands.append(cmd)
        raise RuntimeError("stop here")

    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(seed.run_cells, "_spawn", spawn)
    with pytest.raises(RuntimeError, match="stop here"):
        seed.main()
    cmd = commands[0]
    assert cmd[cmd.index("--ctx-size") + 1] == "3328" and cmd[cmd.index("--parallel") + 1] == "1"


def test_a_concurrency_1_cell_skips_the_pre_check(tmp_path, monkeypatch):
    starts, runs, snapshots = _served(monkeypatch, tmp_path, [_vllm_cap(30240)])
    status = run_cells.run_one(_loaded(concurrency=1), _VERSIONS, tmp_path, _BUDGET)
    assert status["outcome"] == "ok" and runs == [None]
    assert status["kv_capacity_tokens"] == [30240]


def test_main_lists_are_the_generators_output_in_its_seeded_order():
    # docs/adr/0007: the order is drawn once from SEED and is part of the record,
    # so a hand-edited or re-drawn list must fail here.
    spec = importlib.util.spec_from_file_location("make_main_cells",
                                                  _ROOT / "scripts" / "gpu" / "make_main_cells.py")
    make = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make)
    assert make.SEED == 20261024
    for name, cells in make.lists().items():
        committed = (_ROOT / "scripts" / "gpu" / "cells" / name).read_text(encoding="utf-8")
        assert committed == make.render(cells), name
        assert len({c["name"] for c in cells}) == len(cells)
