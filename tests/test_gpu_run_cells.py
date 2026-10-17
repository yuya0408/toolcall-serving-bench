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
