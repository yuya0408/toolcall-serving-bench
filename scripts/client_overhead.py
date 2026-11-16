"""How much the harness's own client adds to TTFT (docs/adr/0003 A8): per
latency cell, the engine's own TTFT and end-to-end time (/metrics) against the
client's records, and whether taking a fixed time off every request's TTFT
would change any verdict.

    python scripts/client_overhead.py runs/            # Release runs-2026-10 unpacked

vLLM and SGLang only: their /metrics keep TTFT and end-to-end histograms over
the whole cell, and each cell starts a fresh server, so the engine's request
count must equal the client's (cells where it does not are skipped and listed).
TensorRT-LLM clears its statistics on every read, and llama.cpp reports no
TTFT histogram. Only means are compared: /metrics keeps sums, not requests.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from eval_bench.latency import SLO_CLASSES, verdict  # noqa: E402

_HIST = {
    "vllm": (r'^vllm:time_to_first_token_seconds_(sum|count)\{[^}]*\} (\S+)',
             r'^vllm:e2e_request_latency_seconds_(sum|count)\{[^}]*\} (\S+)'),
    "sglang": (r'^sglang:time_to_first_token_seconds_(sum|count)\{[^}]*is_streaming="true"[^}]*\} (\S+)',
               r'^sglang:e2e_request_latency_seconds_(sum|count)\{[^}]*is_streaming="true"[^}]*\} (\S+)'),
}
_LIMITS = SLO_CLASSES["conversational"]


def histogram(text: str, pattern: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in re.finditer(pattern, text, re.M):
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def raw_rows(cell: pathlib.Path) -> list[dict] | None:
    raws = sorted(cell.glob("*.raw.jsonl"))
    if len(raws) != 1:
        return None
    return [json.loads(line) for line in raws[0].read_text(encoding="utf-8").splitlines()]


def exceeded(rows: list[dict], shift_ms: float) -> int:
    """The conversational exceedance count (eval_bench.latency.exceeds) with
    shift_ms taken off every TTFT."""
    k = 0
    for r in rows:
        if r["status"] == "timeout" or r["ttft_ms"] is None or r["ttft_ms"] - shift_ms > _LIMITS["ttft_ms"]:
            k += 1
        elif r.get("tpot_ms") is not None and r["tpot_ms"] > _LIMITS["tpot_ms"]:
            k += 1
    return k


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", type=pathlib.Path)
    p.add_argument("--shift-ms", type=float, default=30.0,
                   help="time taken off every TTFT for the verdict check (default 30)")
    args = p.parse_args(argv)

    print(f"{'cell':40} {'n':>5} | {'TTFT eng':>8} {'client':>8} {'diff':>5} | {'e2e eng':>8} {'client':>8} {'diff':>5}")
    skipped, diffs = [], []
    for cell in sorted(args.runs.glob("*/lat-*")):
        engine = cell.name.split("-")[1]
        rows = raw_rows(cell)
        after = cell / "engine-metrics-after.txt"
        if engine not in _HIST or rows is None or not after.exists():
            continue
        ttft, e2e = (histogram(after.read_text(encoding="utf-8"), pat) for pat in _HIST[engine])
        ok = [r for r in rows if r["status"] == "ok"]
        if len(ok) != len(rows) or ttft.get("count") != len(ok) or e2e.get("count") != len(ok):
            skipped.append(f"{cell.parent.name}/{cell.name}")
            continue
        means = (ttft["sum"] / ttft["count"] * 1000, sum(r["ttft_ms"] for r in ok) / len(ok),
                 e2e["sum"] / e2e["count"] * 1000, sum(r["total_ms"] for r in ok) / len(ok))
        diffs.append(means[1] - means[0])
        print(f"{cell.parent.name[:16] + '/' + cell.name:40} {len(ok):5d} | {means[0]:8.1f} {means[1]:8.1f} "
              f"{means[1] - means[0]:5.1f} | {means[2]:8.1f} {means[3]:8.1f} {means[3] - means[2]:5.1f}")
    if diffs:
        print(f"\nclient - engine TTFT, mean per cell: {min(diffs):.1f} to {max(diffs):.1f} ms over {len(diffs)} cells")
    if skipped:
        print("skipped (counts differ or a request did not finish ok):", ", ".join(skipped))

    changed, n_cells = [], 0
    for cell in sorted(args.runs.glob("*/lat-*")):
        rows = raw_rows(cell)
        measured = [r for r in rows or [] if r.get("phase") == "measured"]
        if not measured:
            continue
        n_cells += 1
        before = verdict(exceeded(measured, 0.0), len(measured))["verdict"]
        after_shift = verdict(exceeded(measured, args.shift_ms), len(measured))["verdict"]
        if before != after_shift:
            changed.append(f"{cell.parent.name}/{cell.name}: {before} -> {after_shift}")
    print(f"\nconversational verdicts with {args.shift_ms:g} ms off every TTFT: "
          f"{len(changed)} of {n_cells} cells change" + (": " + "; ".join(changed) if changed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
