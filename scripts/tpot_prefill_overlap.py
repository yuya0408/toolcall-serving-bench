"""Group each request's TPOT by how many other requests' prefills overlapped its
generation (ADR-0009 A5: why vLLM FP8's TPOT rose at concurrency 4).

    python scripts/tpot_prefill_overlap.py runs/gpu-20261028-i-0bd6a9b5da22d5bf5 \
        lat-vllm-fp8-c4 lat-trt-fp8-c4 lat-vllm-bf16-c4 lat-trt-bf16-c4

The runs/ logs are not in the repository; unpack Release runs-2026-10 into runs/
to rerun it. Only the client records (<cell>/*.raw.jsonl) are read:
  generation  sent_at_s + ttft_ms .. done_at_s of a measured request
  prefill     sent_at_s .. sent_at_s + ttft_ms of every other ok request,
              warmup and cooldown included (they share the server with it)

The prefill interval is seen from the client, so it includes queue wait. This
is an association in the records, not a measurement with the overlap varied.
"""
from __future__ import annotations
import argparse
import glob
import json
import statistics

MIN_OUTPUT_TOKENS = 5   # too few tokens for a TPOT worth grouping


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [r for r in (json.loads(line) for line in f) if r["status"] == "ok"]


def overlaps(rows: list[dict]) -> list[tuple[int, float]]:
    """(other prefills overlapping the generation, tpot_ms) per measured request."""
    prefill = [(r["sent_at_s"], r["sent_at_s"] + r["ttft_ms"] / 1000) for r in rows]
    out = []
    for i, r in enumerate(rows):
        if r["phase"] != "measured" or r["output_tokens"] < MIN_OUTPUT_TOKENS:
            continue
        start, end = r["sent_at_s"] + r["ttft_ms"] / 1000, r["done_at_s"]
        if end <= start:
            continue
        n = sum(1 for j, (s, e) in enumerate(prefill) if j != i and min(end, e) > max(start, s))
        out.append((n, r["tpot_ms"]))
    return out


def summarize(pairs: list[tuple[int, float]], tpot_limit_ms: float) -> list[dict]:
    """Per overlap count: requests, median TPOT and how many exceeded the limit."""
    rows = []
    for n in sorted({n for n, _ in pairs}):
        tpot = [t for k, t in pairs if k == n]
        rows.append({"overlap": n, "requests": len(tpot), "tpot_median_ms": statistics.median(tpot),
                     "over_limit": sum(t > tpot_limit_ms for t in tpot)})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("cells", nargs="+", help="cell directories under run_dir, e.g. lat-vllm-fp8-c4")
    ap.add_argument("--tpot-limit-ms", type=float, default=100)   # ADR-0003
    args = ap.parse_args(argv)

    for cell in args.cells:
        files = glob.glob(f"{args.run_dir}/{cell}/*.raw.jsonl")
        if len(files) != 1:
            raise SystemExit(f"{cell}: expected one raw.jsonl, found {len(files)}")
        pairs = overlaps(load(files[0]))
        print(f"== {cell}  measured={len(pairs)}  TPOT median={statistics.median(t for _, t in pairs):.1f}ms")
        for row in summarize(pairs, args.tpot_limit_ms):
            print(f"   overlap={row['overlap']}: requests={row['requests']}  "
                  f"TPOT median={row['tpot_median_ms']:.1f}ms  >{args.tpot_limit_ms:.0f}ms={row['over_limit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
