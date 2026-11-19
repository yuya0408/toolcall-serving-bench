"""Read the prefill-kernel check (docs/prefill-kernel-check.md) by its rules,
offline from the collected folder.

    python scripts/prefill_check_summary.py runs/<dir>

Prints the medians of both rounds, each configuration's verdict against bf16
(and Triton against fp8-block and fp8-tensor) by the "difference" defined in
the document, the ratio at 2,304 tokens, and the profiler's GPU time of the
2,048-token prefill split into linear layers, attention and the rest.
"""
from __future__ import annotations
import argparse
import gzip
import json
import pathlib
import re
import sys

CONFIGS = ("bf16", "fp8-block", "fp8-tensor", "fp8-block-triton", "awq")
LENGTHS = ("16", "64", "128", "256", "512", "1024", "2048", "2304")
DECODE = "decode"


def value(summary: dict, n: str) -> float:
    return summary["decode_step_ms"] if n == DECODE else summary["prefill_ms"][n]


def verdict(s: dict, x: str, y: str, n: str) -> int:
    """+1 x is slower than y, -1 faster, 0 no difference (the document's definition)."""
    d = [value(s[x, r], n) - value(s[y, r], n) for r in (1, 2)]
    if (d[0] > 0) != (d[1] > 0):
        return 0
    drift = max(abs(value(s[c, 1], n) - value(s[c, 2], n)) for c in (x, y))
    slower = x if d[0] > 0 else y
    five_percent = 0.05 * (value(s[slower, 1], n) + value(s[slower, 2], n)) / 2
    return (1 if d[0] > 0 else -1) if min(abs(v) for v in d) > max(drift, five_percent) else 0


def kernel_kind(name: str) -> str:
    if re.search(r"flash_fwd|varlen", name):
        return "attention"
    if name.startswith("triton_"):          # fused elementwise ops (and the activation quantization)
        return "other"
    if re.search(r"gemm|Gemm|[Mm]arlin|_w8a8_triton_block_scaled_mm", name):
        return "linear"
    if "gemvx" in name:                     # lm_head on the last token only
        return "lm_head"
    return "other"


def profile_split(folder: pathlib.Path) -> dict[str, float]:
    trace = next(folder.glob("*.pt.trace.json.gz"))
    split = {"linear": 0.0, "attention": 0.0, "lm_head": 0.0, "other": 0.0}
    for e in json.loads(gzip.open(trace).read())["traceEvents"]:
        if e.get("cat") == "kernel":
            split[kernel_kind(e["name"])] += e["dur"] / 1000
    return split


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=pathlib.Path, help="the collected folder (runs/<dir>)")
    args = p.parse_args(argv)
    d = args.run / "prefill"
    s = {(c, r): json.loads((d / f"{c}-r{r}.json").read_text())["summary"] for c in CONFIGS for r in (1, 2)}

    print("median ms, round 1 / round 2")
    print(f"{'':>8}" + "".join(f"{c:>20}" for c in CONFIGS))
    for n in LENGTHS + (DECODE,):
        print(f"{n:>8}" + "".join(f"{value(s[c, 1], n):>11.1f} /{value(s[c, 2], n):>6.1f}" for c in CONFIGS))

    print("\nverdict (+1 slower, -1 faster, 0 no difference) and ratio at 2,304")
    pairs = [(c, "bf16") for c in CONFIGS[1:]] + [("fp8-block-triton", "fp8-block"), ("fp8-block-triton", "fp8-tensor")]
    for x, y in pairs:
        cells = " ".join(f"{n}:{verdict(s, x, y, n):+d}" for n in LENGTHS + (DECODE,))
        ratio = sum(value(s[x, r], "2304") / value(s[y, r], "2304") for r in (1, 2)) / 2
        print(f"{x} vs {y}: {cells}  ratio {ratio:.3f}")

    print("\nprofiler, 2,048-token prefill, GPU ms (round 1 / round 2)")
    for c in CONFIGS:
        a, b = (profile_split(d / "profile" / f"{c}-r{r}") for r in (1, 2))
        cells = "  ".join(f"{k} {a[k]:.1f}/{b[k]:.1f}" for k in a)
        print(f"{c:>17}  {cells}  total {sum(a.values()):.1f}/{sum(b.values()):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
