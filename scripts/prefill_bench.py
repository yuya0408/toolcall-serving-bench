"""Time one request's prefill against its length, for one vLLM configuration
(docs/prefill-kernel-check.md: why vLLM's W8A16 FP8 prefills slower than bf16 on L4).

    python scripts/prefill_bench.py --name fp8-block --out ~/out/prefill \
        --engine '{"model": "Qwen/Qwen3-8B-FP8", "revision": "220b46e..."}'

One process per configuration, so each starts on a free GPU. Every request is
sent alone (the prefill's tokens are the whole batch, M = prompt length up to
the 2,048-token chunk), with random token ids so that no two prompts share a
prefix, and prefix caching off. The time is the wall time of llm.generate()
for one prompt: with max_tokens=1 it is the prefill plus one sampling step;
with max_tokens=1+DECODE_TOKENS, the difference over DECODE_TOKENS is the time
of one decode step (M = 1).

Writes <out>/<name>.json (every repetition) and prints the medians.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

LENGTHS = (16, 64, 256, 1024, 2048, 2304)   # 2,304: the median prompt (about 2,300) runs as two chunks
WARMUP, REPS = 3, 10
DECODE_TOKENS = 64
DECODE_LENGTH = 2048
# The same engine settings as the phase-2 vLLM cells, except prefix caching
# (off, so that repeated lengths are not served from the cache).
BASE_ENGINE = {"dtype": "bfloat16", "max_model_len": 4096, "max_num_batched_tokens": 2048,
               "gpu_memory_utilization": 0.85, "enable_prefix_caching": False, "seed": 0}


def summarize(rows: list[dict]) -> dict:
    """{"prefill_ms": {length: median}, "decode_step_ms": median} from the raw repetitions."""
    prefill = {}
    for n in sorted({r["length"] for r in rows if r["kind"] == "prefill"}):
        prefill[n] = statistics.median(r["ms"] for r in rows if r["kind"] == "prefill" and r["length"] == n)
    long = [r["ms"] for r in rows if r["kind"] == "decode"]
    decode = None
    if long and DECODE_LENGTH in prefill:
        decode = (statistics.median(long) - prefill[DECODE_LENGTH]) / DECODE_TOKENS
    return {"prefill_ms": prefill, "decode_step_ms": decode}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--engine", required=True, help="JSON of vllm.LLM arguments, merged over the base settings")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from vllm import LLM, SamplingParams   # only on the GPU instance

    engine = {**BASE_ENGINE, **json.loads(args.engine)}
    llm = LLM(**engine)
    vocab = llm.get_tokenizer().vocab_size   # ids below this are ordinary tokens, never special ones
    rng = random.Random(0)

    def timed(length: int, max_tokens: int) -> float:
        prompt = {"prompt_token_ids": [rng.randrange(1000, vocab) for _ in range(length)]}
        params = SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True)
        start = time.perf_counter()
        llm.generate([prompt], params, use_tqdm=False)
        return (time.perf_counter() - start) * 1000

    rows = []
    for length in LENGTHS:
        for i in range(WARMUP + REPS):
            ms = timed(length, 1)
            if i >= WARMUP:
                rows.append({"kind": "prefill", "length": length, "ms": ms})
    for i in range(WARMUP + REPS):
        ms = timed(DECODE_LENGTH, 1 + DECODE_TOKENS)
        if i >= WARMUP:
            rows.append({"kind": "decode", "length": DECODE_LENGTH, "ms": ms})

    summary = summarize(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.name}.json").write_bytes(json.dumps(
        {"name": args.name, "engine": engine, "rows": rows, "summary": summary}, indent=1).encode())
    print(json.dumps({"name": args.name, **summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
