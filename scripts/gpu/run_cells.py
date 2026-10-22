"""Run a list of cells on the GPU instance, one engine server per cell.

    ~/venv/bin/python scripts/gpu/run_cells.py scripts/gpu/cells/smoke.json

For each cell in the list (a JSON array, see scripts/gpu/cells/), this
starts the engine's server, waits until it answers, runs scripts/run_cell.py
with the cell's arguments, and stops the server. transformers_hf needs no
server: run_cell.py loads the model in-process.

Every cell gets a fresh server, for every engine. llama.cpp has no API that
empties its prompt cache (docs/adr/0003), so it must be restarted between
cells; doing the same for vLLM costs a minute or two per cell and gives each
cell its own startup log. run_cell.py still resets vLLM's prefix cache before
a latency cell.

Memory is sized to the requirement, never searched (docs/adr/0005 1): vLLM
keeps its default --gpu-memory-utilization; llama.cpp gets
--ctx-size = concurrency x the longest request (schema/prompt_budget.json +
max_tokens), --kv-unified and --fit off. Before a concurrency>1 cell runs, the
capacity pre-check reads what the server actually holds (vLLM /metrics;
llama.cpp reports no total, so its --ctx-size, with /props n_ctx as the
per-slot limit); a vLLM startup that comes out short is restarted once
(some first startups on a fresh instance came out exactly 1 GiB short), and a
cell that still does not fit is recorded unsupported without running.

Writes to --out (default ~/out), which scripts/gpu/local.sh collect copies
back while the run goes on:
  <cell name>/            the cell's manifest, raw log, server log and capacity_check.json,
                          and the engine's state around the run (engine-*, see snapshot_engine_state)
  status.jsonl            one line per finished cell: outcome, exit code, timing
  RUNNING / DONE / FAILED markers; DONE names the cell list that finished,
                          FAILED says why the list stopped early

A failed cell (server never ready, run_cell.py error, past its time limit)
is recorded and the list moves on: a configuration that cannot run is a
result (ADR-0003's "unsupported"), and the remaining cells do not depend on
it. The list stops early -- FAILED -- only when going on would waste the
rest: the GPU is not freed after a cell, or this script itself fails.
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
import traceback
from datetime import datetime, timezone
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from eval_bench import capacity  # noqa: E402
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
# run_cell.py's --max-tokens default: the requirement's output half (ADR-0005 1).
MAX_TOKENS = 512
# Server settings a cell may set. llama.cpp's context, slots and unified KV are
# derived from the cell's concurrency (llama_server_settings), never written by hand.
SERVER_KEYS = {"vllm": {"max_model_len"}, "llama_cpp": set()}
# A cell's wall-clock limit unless it sets timeout_min: catches a hung server or
# cell before it holds the list until the auto-shutdown. Not a measured
# duration -- set timeout_min on cells expected to run longer.
DEFAULT_TIMEOUT_MIN = 90
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
        unknown = set(c) - set(REQUIRED) - set(RUN_CELL_ARGS) - {"server", "timeout_min"}
        if unknown:
            raise SystemExit(f"{c['name']}: unknown fields {sorted(unknown)}")
        if c["prefix"] not in ("smoke-", "pilot-", ""):
            raise SystemExit(f"{c['name']}: prefix must be smoke-, pilot- or empty")
        if c["quantization"] not in MODELS:
            raise SystemExit(f"{c['name']}: unknown quantization {c['quantization']}")
        if (c["engine"] == "llama_cpp") != (c["quantization"] == "gguf-bf16"):
            raise SystemExit(f"{c['name']}: llama_cpp runs the GGUF, and only llama_cpp does")
        # run_cell.py's default is 5; consistency needs at least two trials per case
        if c["pass_type"] != "latency" and c.get("n", 5) < 2:
            raise SystemExit(f"{c['name']}: accuracy passes need n >= 2")
        extra = set(c.get("server", {})) - SERVER_KEYS.get(c["engine"], set())
        if extra:
            raise SystemExit(f"{c['name']}: server settings {sorted(extra)} are not set by hand for "
                             f"{c['engine']} (docs/adr/0005 1)")


def llama_server_settings(cell: dict[str, Any], max_prompt_tokens: int) -> dict[str, int]:
    """--parallel and --ctx-size for a llama.cpp cell: one slot per concurrent
    request, sharing one KV cache sized to exactly the requirement."""
    concurrency = cell.get("concurrency", 1)
    per_request = capacity.per_request_max(max_prompt_tokens, MAX_TOKENS)
    return {"parallel": concurrency, "ctx_size": capacity.llama_ctx_size(per_request, concurrency)}


def server_command(cell: dict[str, Any], max_prompt_tokens: int | None = None) -> list[str] | None:
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
        if max_prompt_tokens is None:
            raise ValueError("a llama.cpp server is sized from the prompt budget")
        settings = llama_server_settings(cell, max_prompt_tokens)
        # --fit off: the default (on) may quietly shrink unset arguments to fit memory.
        # -lv 4 logs the settings llama.cpp resolved (flash attention, n_batch,
        # n_ubatch, the KV layout), which -lv 3 leaves out; --metrics opens /metrics.
        return [str(LLAMA_SERVER), "-m", str(GGUF), "--jinja", "-ngl", "999",
                "--host", "127.0.0.1", "--port", str(PORT["llama_cpp"]),
                "--ctx-size", str(settings["ctx_size"]), "--parallel", str(settings["parallel"]),
                "--kv-unified", "--fit", "off", "-lv", "4", "--metrics"]
    return None


def engine_version(engine: str, versions: dict[str, Any]) -> str:
    return {"vllm": versions["vllm"], "llama_cpp": versions["llama_cpp_commit"],
            "transformers_hf": versions["transformers"]}[engine]


def run_cell_command(cell: dict[str, Any], version: str, cell_dir: pathlib.Path,
                     capacity_check: pathlib.Path | None) -> list[str]:
    cmd = [str(VENV_BIN / "python"), str(_ROOT / "scripts" / "run_cell.py"),
           "--model", MODELS[cell["quantization"]], "--engine-version", version,
           "--run-id-prefix", cell["prefix"], "--out-dir", str(cell_dir)]
    for key in RUN_CELL_ARGS:
        if key in cell:
            cmd += ["--" + key.replace("_", "-"), str(cell[key])]
    if cell["engine"] in PORT:
        cmd += ["--base-url", f"http://127.0.0.1:{PORT[cell['engine']]}"]
    if capacity_check is not None:
        cmd += ["--capacity-check", str(capacity_check)]
    return cmd


def needs_capacity_check(cell: dict[str, Any]) -> bool:
    return cell["engine"] in PORT and cell.get("concurrency", 1) > 1


def read_capacity(engine: str, ctx_size: int | None = None) -> dict[str, Any]:
    base = f"http://127.0.0.1:{PORT[engine]}"
    if engine == "vllm":
        return capacity.vllm_capacity(capacity.fetch_metrics(base))
    return capacity.llama_capacity(capacity.fetch_props(base), ctx_size)


def snapshot_engine_state(engine: str, cell_dir: pathlib.Path, tag: str) -> list[str]:
    """Save what the server reports about itself, before and after the run, so
    a cell can be explained afterwards from the engine's side as well as from
    the client's: /metrics for both engines (the difference is the run's
    scheduler counters), and llama.cpp's /props and /slots (the settings it
    resolved). Best effort: a failed read is returned, never raised, since the
    cell's numbers do not depend on it."""
    import requests
    base = f"http://127.0.0.1:{PORT[engine]}"
    paths = {"metrics": "/metrics"}
    if engine == "llama_cpp" and tag == "before":
        paths.update(props="/props", slots="/slots")
    failed = []
    for name, path in paths.items():
        suffix = "txt" if name == "metrics" else "json"
        try:
            response = requests.get(base + path, timeout=30)
            response.raise_for_status()
            (cell_dir / f"engine-{name}-{tag}.{suffix}").write_text(response.text)
        except requests.RequestException as exc:
            failed.append(f"{path} ({tag}): {exc}")
    return failed


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


def _spawn(cmd: list[str], log, env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start cmd in its own process group, so stopping it also stops what it
    started (vLLM's engine-core workers)."""
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=hasattr(os, "killpg"))


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, sig)
        elif proc.poll() is None:
            proc.kill()
    except ProcessLookupError:
        pass


def _stop(proc: subprocess.Popen) -> None:
    """SIGINT and a minute to shut down cleanly, then SIGKILL the whole group:
    a worker left behind would hold VRAM and fail every later cell."""
    if proc.poll() is None:
        _signal_group(proc, signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
    _signal_group(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
    proc.wait()


def _wait_gpu_free(limit_mib: int = 1024, timeout_s: int = 120) -> bool:
    """Whether the GPU is (nearly) empty again: the next server must not start
    while the last one still holds VRAM."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip().splitlines()
        if used and int(used[0]) < limit_mib:
            return True
        time.sleep(3)
    return False


def _start_server(cell: dict[str, Any], server_cmd: list[str], server_log: pathlib.Path,
                  status: dict[str, Any]) -> tuple[subprocess.Popen, bool]:
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"} if cell["engine"] == "vllm" else None
    with server_log.open("w") as log:
        server = _spawn(server_cmd, log, env)
    started = time.monotonic()
    status["server_started_at"] = _now()
    ready = _wait_ready(cell["engine"], server)
    # What restarting the server for every cell costs (docs/gpu-runbook.md).
    status["server_ready_s"] = round(time.monotonic() - started, 1)
    return server, ready


def run_one(cell: dict[str, Any], versions: dict[str, Any], out: pathlib.Path,
            budget: dict[str, Any] | None = None) -> dict[str, Any]:
    cell_dir = out / cell["name"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"name": cell["name"], "started_at": _now()}
    max_prompt = budget["max_prompt_tokens"] if budget else None
    server_cmd = server_command(cell, max_prompt)
    server_log = cell_dir / "server.log" if server_cmd else None
    timeout_s = 60 * cell.get("timeout_min", DEFAULT_TIMEOUT_MIN)
    server = runner = None
    check_path = None
    try:
        if server_cmd:
            (cell_dir / "server_command.txt").write_text(" ".join(server_cmd) + "\n")
            restarts = 0
            while True:
                server, ready = _start_server(cell, server_cmd, server_log, status)
                if not ready:
                    status.update(outcome="server_not_ready", returncode=server.poll())
                    return status
                if cell["engine"] == "vllm":
                    # Every vLLM startup's capacity, to see whether the short ones recur (ADR-0005).
                    status.setdefault("kv_capacity_tokens", []).append(read_capacity("vllm")["tokens"])
                if not needs_capacity_check(cell):
                    break
                ctx_size = (llama_server_settings(cell, max_prompt)["ctx_size"]
                            if cell["engine"] == "llama_cpp" else None)
                pre = capacity.pre_check(cell["engine"], read_capacity(cell["engine"], ctx_size),
                                         max_prompt, MAX_TOKENS, cell["concurrency"])
                if pre["fits"] or cell["engine"] != "vllm" or restarts == 1:
                    break
                restarts += 1            # a short vLLM startup: restart once (ADR-0005 2)
                _stop(server)
                server = None
                # Keep the short startup's log: why some startups come out short is still open.
                server_log.replace(cell_dir / "server-short-startup.log")
                if not _wait_gpu_free():
                    status.update(outcome="error", reason="GPU not freed for the capacity restart")
                    return status
            if needs_capacity_check(cell):
                pre["restarts"] = restarts
                pre["startup_log_max_concurrency"] = (
                    capacity.startup_log_max_concurrency(server_log.read_text(errors="replace"))
                    if cell["engine"] == "vllm" else None)
                check_path = cell_dir / "capacity_check.json"
                check_path.write_text(json.dumps(pre, indent=2) + "\n")
                status["capacity_fits"] = pre["fits"]
                if not pre["fits"]:
                    # ADR-0003's "unsupported": a result, recorded without running.
                    status.update(outcome="unsupported", reason="capacity pre-check (docs/adr/0005)")
                    return status
        if server is not None:
            status["engine_state_failed"] = snapshot_engine_state(cell["engine"], cell_dir, "before")
        cmd = run_cell_command(cell, engine_version(cell["engine"], versions), cell_dir, check_path)
        (cell_dir / "run_cell_command.txt").write_text(" ".join(cmd) + "\n")
        status["run_started_at"] = _now()
        with (cell_dir / "run_cell.log").open("w") as log:
            runner = _spawn(cmd, log)
        try:
            rc = runner.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # A hung server or cell would otherwise hold the list until the auto-shutdown.
            status.update(outcome="timeout", returncode=None, timeout_min=timeout_s / 60)
            return status
        # run_cell.py exits 2 for an aborted latency cell, which it records as "unsupported",
        # and 3 when the capacity post-check failed (the manifest is kept, not used for a verdict).
        status.update(outcome={0: "ok", 2: "unsupported", 3: "capacity_misestimate"}.get(rc, "error"),
                      returncode=rc)
        return status
    finally:
        if runner is not None and server is not None and server.poll() is None:
            status.setdefault("engine_state_failed", []).extend(
                snapshot_engine_state(cell["engine"], cell_dir, "after"))
        for proc in (runner, server):
            if proc is not None:
                _stop(proc)
        status["gpu_freed"] = _wait_gpu_free()
        status["finished_at"] = _now()


def load_budget(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The counted longest prompt (schema/prompt_budget.json), checked against
    the catalog and the whole case set: a cell running a subset (max_cases)
    stays within the set's longest prompt."""
    stages = {c["tool_stage"] for c in cells if c["engine"] in PORT}
    if not stages:
        return None
    if len(stages) > 1:
        raise SystemExit("one cell list per tool_stage: the prompt budget is per stage")
    from eval_bench.dataset.loader import load_cases
    stage = stages.pop()
    catalog = _ROOT / "schema" / ("tools.json" if stage == 10 else "tools_extended.json")
    version = json.loads(catalog.read_text(encoding="utf-8"))["version"]
    return capacity.load_prompt_budget(stage, version, load_cases(_ROOT / "data"))


def swap_devices(proc_swaps: pathlib.Path = pathlib.Path("/proc/swaps")) -> list[str]:
    """Active swap areas. setup.sh uses swap only while it builds and converts;
    a cell measured with swap on could be slowed by paging, so none may start."""
    if not proc_swaps.exists():
        return []
    return [line.split()[0] for line in proc_swaps.read_text().splitlines()[1:] if line.strip()]


def _mark(out: pathlib.Path, name: str, text: str | None) -> None:
    path = out / name
    if text is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cells", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=HOME / "out")
    ap.add_argument("--only", nargs="*", help="run just these cell names")
    args = ap.parse_args(argv)

    cells = json.loads(args.cells.read_text(encoding="utf-8"))
    validate(cells)
    # Tools the engines shell out to (FlashInfer's JIT runs ninja) live in the
    # venv, which a non-login SSH shell does not have on PATH.
    os.environ["PATH"] = f"{VENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
    if args.only:
        cells = [c for c in cells if c["name"] in set(args.only)]

    for name in ("DONE", "FAILED"):
        _mark(args.out, name, None)
    _mark(args.out, "RUNNING", f"{args.cells.name} {_now()}\n")
    failure = None
    try:
        versions = json.loads((args.out / "versions.json").read_text())   # written by setup.sh
        budget = load_budget(cells)
        swaps = swap_devices()
        if swaps:
            failure = (f"swap is active ({', '.join(swaps)}); paging would distort the measurements. "
                       "Turn it off (sudo swapoff -a) and start the list again")
            cells = []
        for cell in cells:
            print(f"[{_now()}] {cell['name']}", flush=True)
            status = run_one(cell, versions, args.out, budget)
            print(f"[{_now()}] {cell['name']}: {status['outcome']}", flush=True)
            with (args.out / "status.jsonl").open("a") as f:
                f.write(json.dumps(status) + "\n")
            if not status["gpu_freed"]:
                failure = (f"GPU memory still in use after {cell['name']}; the remaining cells "
                           "would all fail to start, so the list stops here")
                break
    except BaseException:
        failure = traceback.format_exc()
        raise
    finally:
        _mark(args.out, "RUNNING", None)
        if failure:
            _mark(args.out, "FAILED", f"{args.cells.name} {_now()}\n{failure}\n")
            print(f"FAILED: {failure}", flush=True)
        else:
            _mark(args.out, "DONE", f"{args.cells.name} {_now()}\n")
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
