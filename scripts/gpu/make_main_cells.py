"""Write the main run's cell lists (docs/adr/0007, docs/adr/0008).

    python scripts/gpu/make_main_cells.py

- cells/main-latency.json: every latency cell, on one instance, so every
  speed comparison shares one physical GPU.
- cells/main-accuracy.json: the plain accuracy cells and the third pass
  (the concurrency 1 / 16 pairs), on a second instance.
- cells/main-quant.json (docs/adr/0008): the quantization non-inferiority
  cells (bf16 and 4bit of each engine, greedy, concurrency 1), llama.cpp
  Q4_K_M's latency levels, and vLLM AWQ concurrency 4 again to see how far
  this instance's speed is from main-latency's.
- cells/main-batching.json (docs/adr/0009): vLLM, SGLang and TensorRT-LLM,
  each in bf16, 4-bit and FP8, at concurrency 1-16, on one instance; then
  the accuracy cells (the new engines' plain accuracy, and the greedy cells
  for the bf16 gate and the quantization non-inferiority). The latency cells
  come first so an interruption costs as little of them as possible
  (docs/adr/0009 7); each block is shuffled from BATCHING_SEED.
- cells/smoke-batching.json: the same configurations, short, to check each
  engine against docs/adr/0009 2 before the main list.

Each list runs in a random order drawn once from SEED (main-quant.json from
QUANT_SEED, so adding it left the first two lists as they were run), fixed
before that list's results exist: a time trend on the instance (interference, heat) then lands
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
QUANT_SEED = 20261026
BATCHING_SEED = 20261027
CELLS_DIR = pathlib.Path(__file__).resolve().parent / "cells"
DEFAULT_TIMEOUT_MIN = 90   # run_cells.py's

CONFIGS = {   # name -> (engine, quantization)
    "vllm-bf16": ("vllm", "bf16-none"),
    "vllm-awq": ("vllm", "awq-4bit"),
    "llama-gguf": ("llama_cpp", "gguf-bf16"),
    "hf-bf16": ("transformers_hf", "bf16-none"),
    "llama-q4km": ("llama_cpp", "gguf-q4_k_m"),
    "vllm-fp8": ("vllm", "fp8-block"),                    # docs/adr/0009
    "sglang-bf16": ("sglang", "bf16-none"),
    "sglang-awq": ("sglang", "awq-4bit"),
    "sglang-fp8": ("sglang", "fp8-block"),
    "trt-bf16": ("tensorrt_llm", "bf16-none"),
    "trt-awq": ("tensorrt_llm", "modelopt-int4-awq"),
    "trt-fp8": ("tensorrt_llm", "modelopt-fp8"),
}
PASS_TYPES = {"acc": "accuracy", "third": "accuracy_under_load", "ni": "accuracy_under_load",
              "lat": "latency"}

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
# docs/adr/0008. Non-inferiority ("ni"): each engine's bf16 and 4bit, the third
# pass's settings at concurrency 1 (greedy, same tool orders), all re-measured
# on this instance. llama.cpp Q4_K_M has no measurement yet: its estimates are
# guesses, the accuracy cell set to bf16's so its timeout is not too short.
QUANT = [
    ("ni", "vllm-bf16", 1, 74), ("ni", "vllm-awq", 1, 35),
    ("ni", "llama-gguf", 1, 78), ("ni", "llama-q4km", 1, 78),
    ("lat", "llama-q4km", 1, 70), ("lat", "llama-q4km", 2, 55), ("lat", "llama-q4km", 4, 45),
    ("lat", "llama-q4km", 8, 40), ("lat", "llama-q4km", 16, 35), ("lat", "llama-q4km", 32, 35),
    ("lat", "vllm-awq", 4, 28),
]

# docs/adr/0009. Estimated minutes per latency level, from main-latency's
# measured throughput (1,200 requests / requests per second, plus ~4 min for the
# server start): bf16 from vLLM bf16 (0.244-0.896 req/s), 4-bit from vLLM AWQ
# (0.557-1.264); FP8 is a guess between the two (the geometric mean). SGLang
# and TensorRT-LLM are assumed as fast as vLLM. bf16 at 16 was unsupported for
# vLLM, but may fit another engine, so it is given a full cell's time.
_LEVEL_MIN = {"bf16": {1: 86, 2: 54, 4: 35, 8: 26, 16: 20},
              "awq": {1: 40, 2: 30, 4: 24, 8: 21, 16: 20},
              "fp8": {1: 58, 2: 40, 4: 29, 8: 23, 16: 22}}
_ENGINES = ("vllm", "sglang", "trt")
BATCHING_LATENCY = [("lat", f"{e}-{q}", c, m) for e in _ENGINES for q, levels in _LEVEL_MIN.items()
                    for c, m in levels.items()]
# Plain accuracy for the new engines' bf16 (vLLM's is main-accuracy's); greedy
# concurrency 1 for all nine configs, vLLM's included: the gate pools the bf16s
# of this instance (docs/adr/0009 5). Minutes from main-accuracy / main-quant
# (acc-vllm-bf16 74, ni-vllm-bf16 74, ni-vllm-awq 32); FP8 between.
BATCHING_ACCURACY = [
    ("acc", "sglang-bf16", 1, 74), ("acc", "trt-bf16", 1, 74),
    ("ni", "vllm-bf16", 1, 75), ("ni", "vllm-awq", 1, 35), ("ni", "vllm-fp8", 1, 55),
    ("ni", "sglang-bf16", 1, 75), ("ni", "sglang-awq", 1, 35), ("ni", "sglang-fp8", 1, 55),
    ("ni", "trt-bf16", 1, 75), ("ni", "trt-awq", 1, 35), ("ni", "trt-fp8", 1, 55),
]
# Smoke (docs/adr/0009 2): per configuration, two identical short greedy cells
# (does greedy repeat itself; the tool-call shape; prompt_tokens against vLLM's)
# and short latency cells at concurrency 1 and 16 (capacity pre-check, the
# post-check counters, streamed usage); per engine, one short sampled accuracy
# cell on bf16 (the sampling knobs are the engine's, not the checkpoint's).
SMOKE_BATCHING = ([("acc", f"{e}-bf16", 1, "") for e in _ENGINES]
                  + [(k, f"{e}-{q}", c, r) for e in _ENGINES for q in _LEVEL_MIN
                     for k, c, r in (("ni", 1, ""), ("ni", 1, "-r2"), ("lat", 1, ""), ("lat", 16, ""))])


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
    if kind in ("third", "ni"):
        c["temperature"] = 0     # greedy (docs/adr/0002, docs/adr/0008)
    if engine in ("vllm", "sglang", "tensorrt_llm"):
        c["server"] = {"max_model_len": 8192}   # the same context for every batching engine (docs/adr/0009 5)
    timeout = max(DEFAULT_TIMEOUT_MIN, math.ceil(est_min * 1.5 / 10) * 10)
    if timeout > DEFAULT_TIMEOUT_MIN:
        c["timeout_min"] = timeout
    return c


def ordered(entries: list[tuple], rng: random.Random) -> list[dict]:
    cells = [cell(*e) for e in entries]
    rng.shuffle(cells)
    return cells


def smoke(kind: str, config: str, concurrency: int, repeat: str) -> dict:
    c = cell(kind, config, concurrency, 0)
    c.update(name=c["name"] + repeat, prefix="smoke-", max_cases=20)
    if kind == "lat":
        c["n_requests"] = 60
    else:
        c["n"] = 2
    return c


def render(cells: list[dict]) -> str:
    return "[\n" + ",\n".join("  " + json.dumps(c, ensure_ascii=False) for c in cells) + "\n]\n"


def lists() -> dict[str, list[dict]]:
    rng = random.Random(SEED)   # one generator, drawn in this order: latency, then accuracy
    batching = random.Random(BATCHING_SEED)   # latency block first, then accuracy
    return {"main-latency.json": ordered(LATENCY, rng), "main-accuracy.json": ordered(ACCURACY, rng),
            "main-quant.json": ordered(QUANT, random.Random(QUANT_SEED)),
            "main-batching.json": ordered(BATCHING_LATENCY, batching) + ordered(BATCHING_ACCURACY, batching),
            "smoke-batching.json": [smoke(*e) for e in SMOKE_BATCHING]}


def estimated_minutes() -> dict[str, int]:
    return {"main-latency.json": sum(e[3] for e in LATENCY),
            "main-accuracy.json": sum(e[3] for e in ACCURACY),
            "main-quant.json": sum(e[3] for e in QUANT),
            "main-batching.json": sum(e[3] for e in BATCHING_LATENCY + BATCHING_ACCURACY),
            "smoke-batching.json": 4 * len(SMOKE_BATCHING)}   # ~4 min each, mostly the server start


if __name__ == "__main__":
    for name, cells in lists().items():
        (CELLS_DIR / name).write_text(render(cells), encoding="utf-8", newline="\n")
        print(f"{name}: {len(cells)} cells, ~{estimated_minutes()[name] / 60:.1f} h estimated")
