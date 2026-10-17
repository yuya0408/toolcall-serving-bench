"""Run a list of cells on the GPU instance, one engine server per cell.

    ~/venv/bin/python scripts/gpu/run_cells.py scripts/gpu/cells/smoke.json

For each cell in the list (a JSON array, see scripts/gpu/cells/), this
starts the engine's server, waits until it answers, runs scripts/run_cell.py
with the cell's arguments, and stops the server. transformers_hf needs no
server: run_cell.py loads the model in-process.

Every cell gets a fresh server, for every engine. llama.cpp has no API that
empties its prompt cache (docs/adr/0003), so it must be restarted between
cells; doing the same for vLLM costs a minute or two per cell and gives each
cell its own startup log, which the queueing guard reads
(eval_bench/runner.py parse_vllm_diagnostics). run_cell.py still resets
vLLM's prefix cache before a latency cell.

Writes to --out (default ~/out), which scripts/gpu/local.sh collect copies
back while the run goes on:
  <cell name>/            the cell's manifest, raw log and server log
  status.jsonl            one line per finished cell: exit code and timing
  RUNNING / DONE          markers; DONE names the cell list that finished

A failed cell (server never ready, run_cell.py error) is recorded and the
list moves on: a configuration that cannot run is a result (ADR-0003's
"unsupported"), and the remaining cells do not depend on it.
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parents[2]
HOME = pathlib.Path.home()
VENV_BIN = HOME / "venv" / "bin"
LLAMA_SERVER = HOME / "llama.cpp" / "build" / "bin" / "llama-server"
GGUF = HOME / "models" / "Qwen3-8B-bf16.gguf"
PORT = {"vllm": 8000, "llama_cpp": 8080}
READY_TIMEOUT_S = {"vllm": 900, "llama_cpp": 300}
MODELS = {   # quantization -> the model id the manifest records and vLLM serves (from the HF cache)
    "bf16-none": "Qwen/Qwen3-8B",
    "awq-4bit": "Qwen/Qwen3-8B-AWQ",
    "gguf-bf16": "Qwen/Qwen3-8B",
}
# Arguments a cell may pass through to run_cell.py, as written in the cell list.
RUN_CELL_ARGS = ("pass_type", "engine", "quantization", "tool_stage", "concurrency", "n",
                 "n_requests", "temperature", "max_cases", "seed_base", "request_timeout")
REQUIRED = ("name", "pass_type", "engine", "quantization", "tool_stage", "prefix")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate(cells: list[dict[str, Any]]) -> None:
    names = [c.get("name") for c in cells]
    if len(set(names)) != len(names):
        raise SystemExit("cell names must be unique: they are the output directories")
    for c in cells:
        missing = [k for k in REQUIRED if k not in c]
        if missing:
            raise SystemExit(f"{c.get('name')}: missing {missing}")
        unknown = set(c) - set(REQUIRED) - set(RUN_CELL_ARGS) - {"server"}
        if unknown:
            raise SystemExit(f"{c['name']}: unknown fields {sorted(unknown)}")
        if c["prefix"] not in ("smoke-", "pilot-", ""):
            raise SystemExit(f"{c['name']}: prefix must be smoke-, pilot- or empty")
        if c["quantization"] not in MODELS:
            raise SystemExit(f"{c['name']}: unknown quantization {c['quantization']}")
        if (c["engine"] == "llama_cpp") != (c["quantization"] == "gguf-bf16"):
            raise SystemExit(f"{c['name']}: llama_cpp runs the GGUF, and only llama_cpp does")


def server_command(cell: dict[str, Any]) -> list[str] | None:
    """The engine server for one cell, or None when the engine runs in-process."""
    server = cell.get("server", {})
    if cell["engine"] == "vllm":
        model = MODELS[cell["quantization"]]
        cmd = [str(VENV_BIN / "vllm"), "serve", model, "--served-model-name", model,
               "--host", "127.0.0.1", "--port", str(PORT["vllm"]),
               "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
               "--enable-prompt-tokens-details"]
        if "max_model_len" in server:
            cmd += ["--max-model-len", str(server["max_model_len"])]
        return cmd
    if cell["engine"] == "llama_cpp":
        cmd = [str(LLAMA_SERVER), "-m", str(GGUF), "--jinja", "-ngl", "999",
               "--host", "127.0.0.1", "--port", str(PORT["llama_cpp"]),
               "--ctx-size", str(server.get("ctx_size", 8192)),
               "--parallel", str(server.get("parallel", 1))]
        if server.get("kv_unified"):
            cmd.append("--kv-unified")
        return cmd
    return None


def engine_version(engine: str, versions: dict[str, Any]) -> str:
    return {"vllm": versions["vllm"], "llama_cpp": versions["llama_cpp_commit"],
            "transformers_hf": versions["transformers"]}[engine]


def run_cell_command(cell: dict[str, Any], version: str, cell_dir: pathlib.Path,
                     server_log: pathlib.Path | None) -> list[str]:
    cmd = [str(VENV_BIN / "python"), str(_ROOT / "scripts" / "run_cell.py"),
           "--model", MODELS[cell["quantization"]], "--engine-version", version,
           "--run-id-prefix", cell["prefix"], "--out-dir", str(cell_dir)]
    for key in RUN_CELL_ARGS:
        if key in cell:
            cmd += ["--" + key.replace("_", "-"), str(cell[key])]
    if cell["engine"] in PORT:
        cmd += ["--base-url", f"http://127.0.0.1:{PORT[cell['engine']]}"]
    if cell["engine"] == "vllm" and server_log is not None:
        cmd += ["--vllm-startup-log", str(server_log)]
    return cmd


def _wait_ready(engine: str, proc: subprocess.Popen) -> bool:
    import requests
    deadline = time.monotonic() + READY_TIMEOUT_S[engine]
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if requests.get(f"http://127.0.0.1:{PORT[engine]}/health", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    _wait_gpu_free()


def _wait_gpu_free(limit_mib: int = 1024, timeout_s: int = 120) -> None:
    """The next server must not start while the last one still holds VRAM."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip().splitlines()
        if used and int(used[0]) < limit_mib:
            return
        time.sleep(3)


def run_one(cell: dict[str, Any], versions: dict[str, Any], out: pathlib.Path) -> dict[str, Any]:
    cell_dir = out / cell["name"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"name": cell["name"], "started_at": _now()}
    server_cmd = server_command(cell)
    server_log = cell_dir / "server.log" if server_cmd else None
    proc = None
    try:
        if server_cmd:
            env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"} if cell["engine"] == "vllm" else None
            (cell_dir / "server_command.txt").write_text(" ".join(server_cmd) + "\n")
            with server_log.open("w") as log:
                proc = subprocess.Popen(server_cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            if not _wait_ready(cell["engine"], proc):
                status.update(outcome="server_not_ready", returncode=proc.poll())
                return status
        cmd = run_cell_command(cell, engine_version(cell["engine"], versions), cell_dir, server_log)
        (cell_dir / "run_cell_command.txt").write_text(" ".join(cmd) + "\n")
        with (cell_dir / "run_cell.log").open("w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        # run_cell.py exits 2 for an aborted latency cell, which it records as "unsupported".
        status.update(outcome={0: "ok", 2: "unsupported"}.get(rc, "error"), returncode=rc)
        return status
    finally:
        if proc is not None:
            _stop(proc)
        status["finished_at"] = _now()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cells", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=HOME / "out")
    ap.add_argument("--only", nargs="*", help="run just these cell names")
    args = ap.parse_args(argv)

    cells = json.loads(args.cells.read_text(encoding="utf-8"))
    validate(cells)
    if args.only:
        cells = [c for c in cells if c["name"] in set(args.only)]
    versions = json.loads((args.out / "versions.json").read_text())   # written by setup.sh

    (args.out / "DONE").unlink(missing_ok=True)
    (args.out / "RUNNING").write_text(f"{args.cells.name} {_now()}\n")
    for cell in cells:
        print(f"[{_now()}] {cell['name']}", flush=True)
        status = run_one(cell, versions, args.out)
        print(f"[{_now()}] {cell['name']}: {status['outcome']}", flush=True)
        with (args.out / "status.jsonl").open("a") as f:
            f.write(json.dumps(status) + "\n")
    (args.out / "RUNNING").unlink(missing_ok=True)
    (args.out / "DONE").write_text(f"{args.cells.name} {_now()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
