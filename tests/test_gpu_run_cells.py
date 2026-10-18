"""scripts/gpu/run_cells.py without a GPU: the cell lists are valid, and each
cell becomes the server and run_cell.py command it should."""
from __future__ import annotations
import importlib.util
import json
import pathlib

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
        log = tmp_path / "server.log" if cell["engine"] == "vllm" else None
        cmd = run_cells.run_cell_command(cell, run_cells.engine_version(cell["engine"], _VERSIONS),
                                         tmp_path, log)
        run_cell._parse_args(cmd[2:])   # the same guards run_cell.py applies on the instance


def test_vllm_server_has_the_flags_the_adrs_require():
    cmd = run_cells.server_command({"engine": "vllm", "quantization": "awq-4bit",
                                    "server": {"max_model_len": 8192}})
    assert cmd[1:3] == ["serve", "Qwen/Qwen3-8B-AWQ"]
    for flag in ("--enable-auto-tool-choice", "--enable-prompt-tokens-details"):
        assert flag in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"
    assert cmd[cmd.index("--max-model-len") + 1] == "8192"


def test_llama_server_runs_the_gguf_with_jinja_and_the_cells_slots():
    cmd = run_cells.server_command({"engine": "llama_cpp", "quantization": "gguf-bf16",
                                    "server": {"ctx_size": 40960, "parallel": 16, "kv_unified": True}})
    assert "--jinja" in cmd and "--kv-unified" in cmd
    assert cmd[cmd.index("--parallel") + 1] == "16"
    assert cmd[cmd.index("-m") + 1].endswith("Qwen3-8B-bf16.gguf")


def test_transformers_hf_needs_no_server_and_vllm_gets_its_startup_log(tmp_path):
    assert run_cells.server_command({"engine": "transformers_hf", "quantization": "bf16-none"}) is None
    cell = {"name": "x", "prefix": "pilot-", "pass_type": "accuracy_under_load", "engine": "vllm",
            "quantization": "awq-4bit", "tool_stage": 10, "concurrency": 16, "n": 5, "temperature": 0}
    cmd = run_cells.run_cell_command(cell, "0.20.0", tmp_path, tmp_path / "server.log")
    assert cmd[cmd.index("--vllm-startup-log") + 1].endswith("server.log")
    assert cmd[cmd.index("--engine-version") + 1] == "0.20.0"
    assert cmd[cmd.index("--model") + 1] == "Qwen/Qwen3-8B-AWQ"


@pytest.mark.parametrize("bad", [
    {"engine": "llama_cpp", "quantization": "bf16-none"},     # llama_cpp runs the GGUF only
    {"engine": "vllm", "quantization": "gguf-bf16"},          # and only llama_cpp runs it
    {"prefix": "dry-"},
    {"servr": {}},                                            # a typo must not be ignored
])
def test_validate_refuses(bad):
    cell = {"name": "x", "prefix": "smoke-", "pass_type": "accuracy", "engine": "vllm",
            "quantization": "bf16-none", "tool_stage": 10, **bad}
    with pytest.raises(SystemExit):
        run_cells.validate([cell])


def _hf_cell(name, **extra):
    return {"name": name, "prefix": "smoke-", "pass_type": "accuracy", "engine": "transformers_hf",
            "quantization": "bf16-none", "tool_stage": 10, "n": 1, **extra}


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
