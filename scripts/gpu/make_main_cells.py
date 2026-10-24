"""Write the main run's two cell lists (docs/adr/0007).

    python scripts/gpu/make_main_cells.py

- cells/main-latency.json: every latency cell, on one instance, so every
  speed comparison shares one physical GPU.
- cells/main-accuracy.json: the plain accuracy cells and the third pass
  (the concurrency 1 / 16 pairs), on a second instance.

Each list runs in a random order drawn once from SEED, fixed before any main
result exists: a time trend on the instance (interference, heat) then lands
on cells at random instead of lining up with one engine (Abedi & Brecht,
ICPE 2017). Do not change SEED to get a different order; the order is part of
the record. tests/test_gpu_run_cells.py checks the committed lists are this
script's output.

The minutes beside each cell are estimates (pilot-1 and pilot-2 throughput;
concurrency 2-8 interpolated geometrically between 1 and 16, vLLM bf16's from
vLLM AWQ's growth). They set each cell's timeout_min (1.5x, at least the
default 90) and the auto-shutdown passed at apply time (docs/gpu-runbook.md).
"""
from __future__ import annotations
import json
import math
import pathlib
import random

SEED = 20261024
CELLS_DIR = pathlib.Path(__file__).resolve().parent / "cells"
DEFAULT_TIMEOUT_MIN = 90   # run_cells.py's

CONFIGS = {   # name -> (engine, quantization)
    "vllm-bf16": ("vllm", "bf16-none"),
    "vllm-awq": ("vllm", "awq-4bit"),
    "llama-gguf": ("llama_cpp", "gguf-bf16"),
    "hf-bf16": ("transformers_hf", "bf16-none"),
}
PASS_TYPES = {"acc": "accuracy", "third": "accuracy_under_load", "lat": "latency"}

# (kind, config, concurrency, estimated minutes including the server start).
# Latency: 1/2/4/8 for every server engine, doubling past 8 up to 32 until a
# level is unsupported (docs/adr/0003 2, G). Cells expected to be unsupported
# stay in the list so the capacity pre-check or the failed start records it
# (docs/adr/0003 H); they take a few minutes.
LATENCY = [
    ("lat", "hf-bf16", 1, 96),
    ("lat", "vllm-bf16", 1, 86), ("lat", "vllm-bf16", 2, 72), ("lat", "vllm-bf16", 4, 60),
    ("lat", "vllm-bf16", 8, 50), ("lat", "vllm-bf16", 16, 5),
    ("lat", "vllm-awq", 1, 40), ("lat", "vllm-awq", 2, 34), ("lat", "vllm-awq", 4, 28),
    ("lat", "vllm-awq", 8, 24), ("lat", "vllm-awq", 16, 20), ("lat", "vllm-awq", 32, 5),
    ("lat", "llama-gguf", 1, 95), ("lat", "llama-gguf", 2, 73), ("lat", "llama-gguf", 4, 57),
    ("lat", "llama-gguf", 8, 44), ("lat", "llama-gguf", 16, 35), ("lat", "llama-gguf", 32, 6),
]
# Plain accuracy at concurrency 1 for every config; the third pass for every
# config that can serve concurrency 16 (docs/adr/0002, 2026-10-19). vLLM bf16's
# concurrency 16 stays so the pre-check records it; transformers_hf has no
# concurrency.
ACCURACY = [
    ("acc", "vllm-bf16", 1, 74), ("acc", "vllm-awq", 1, 35),
    ("acc", "llama-gguf", 1, 76), ("acc", "hf-bf16", 1, 80),
    ("third", "vllm-awq", 1, 35), ("third", "vllm-awq", 16, 17),
    ("third", "llama-gguf", 1, 76), ("third", "llama-gguf", 16, 30),
    ("third", "vllm-bf16", 16, 5),
]


def cell(kind: str, config: str, concurrency: int, est_min: int) -> dict:
    engine, quantization = CONFIGS[config]
    name = f"{kind}-{config}" + ("" if kind == "acc" else f"-c{concurrency}")
    c = {"name": name, "prefix": "", "pass_type": PASS_TYPES[kind], "engine": engine,
         "quantization": quantization, "tool_stage": 10}
    if kind != "acc":
        c["concurrency"] = concurrency
    if kind == "lat":
        c["n_requests"] = 1000   # eval_bench/latency.py MAIN_N_REQUESTS
    else:
        c["n"] = 5
    if kind == "third":
        c["temperature"] = 0     # greedy (docs/adr/0002)
    if engine == "vllm":
        c["server"] = {"max_model_len": 8192}
    timeout = max(DEFAULT_TIMEOUT_MIN, math.ceil(est_min * 1.5 / 10) * 10)
    if timeout > DEFAULT_TIMEOUT_MIN:
        c["timeout_min"] = timeout
    return c


def ordered(entries: list[tuple], rng: random.Random) -> list[dict]:
    cells = [cell(*e) for e in entries]
    rng.shuffle(cells)
    return cells


def render(cells: list[dict]) -> str:
    return "[\n" + ",\n".join("  " + json.dumps(c, ensure_ascii=False) for c in cells) + "\n]\n"


def lists() -> dict[str, list[dict]]:
    rng = random.Random(SEED)   # one generator, drawn in this order: latency, then accuracy
    return {"main-latency.json": ordered(LATENCY, rng), "main-accuracy.json": ordered(ACCURACY, rng)}


def estimated_minutes() -> dict[str, int]:
    return {"main-latency.json": sum(e[3] for e in LATENCY),
            "main-accuracy.json": sum(e[3] for e in ACCURACY)}


if __name__ == "__main__":
    for name, cells in lists().items():
        (CELLS_DIR / name).write_text(render(cells), encoding="utf-8", newline="\n")
        print(f"{name}: {len(cells)} cells, ~{estimated_minutes()[name] / 60:.1f} h estimated")
