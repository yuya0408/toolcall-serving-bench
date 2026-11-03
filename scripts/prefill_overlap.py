"""Split each request's TTFT into queue wait, server prefill and the rest, and
check whether the requests over the TTFT limit overlapped another request's
prefill (README "考察：llama.cpp Q4_K_Mで並行数が増えなかったのは、プリフィルが遅いから").
llama.cpp cells only: vLLM's log has no per-request prefill time.

    python scripts/prefill_overlap.py runs/<dir>/lat-llama-q4km-c2/server.log \
        runs/<dir>/lat-llama-q4km-c2/<run_id>.raw.jsonl

The runs/ logs are not in the repository; unpack Release runs-2026-10 into runs/
(it keeps this cell's server.log) to rerun it. Inputs:
  server.log   llama-server output. Per task it reads the 'processing task'
               timestamp (start), 'new prompt ... task.n_tokens' (prompt length)
               and print_timing's 'prompt eval time' (prefill).
  raw.jsonl    the client records (sent_at_s, ttft_ms, prompt_tokens, phase).

Server tasks and client records are matched by order (send order against start
order); the share whose prompt length agrees is printed as a check. The two
clocks differ, so the smallest start - send gap is taken as zero wait.
No GPU profiler is involved: 'prefill' is llama-server's own timing.
"""
from __future__ import annotations
import argparse
import json
import re
import statistics

# llama-server log lines start with minutes.seconds.milliseconds.microseconds
TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")


def timestamp(line: str) -> float | None:
    m = TS.match(line)
    if not m:
        return None
    mi, s, ms, us = map(int, m.groups())
    return mi * 60 + s + ms / 1e3 + us / 1e6


def server_tasks(log: str) -> list[dict]:
    """Tasks with a start time and a prefill time, in start order."""
    tasks: dict[int, dict] = {}
    for line in open(log, encoding="utf-8", errors="ignore"):
        m = re.search(r"task (\d+) \| processing task", line)
        if m:
            tasks.setdefault(int(m.group(1)), {})["start"] = timestamp(line)
            continue
        m = re.search(r"task (\d+) \| new prompt, .*task\.n_tokens = (\d+)", line)
        if m:
            tasks.setdefault(int(m.group(1)), {})["n"] = int(m.group(2))
            continue
        m = re.search(r"task (\d+) \| prompt eval time =\s*([\d.]+) ms", line)
        if m:
            tasks.setdefault(int(m.group(1)), {})["prefill_ms"] = float(m.group(2))
    out = [t for t in tasks.values() if "start" in t and "prefill_ms" in t]
    out.sort(key=lambda t: t["start"])
    for t in out:
        t["prefill"] = (t["start"], t["start"] + t["prefill_ms"] / 1e3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("server_log")
    ap.add_argument("raw_jsonl")
    ap.add_argument("--ttft-limit-ms", type=float, default=2000)
    args = ap.parse_args()

    tasks = server_tasks(args.server_log)
    rows = sorted((json.loads(l) for l in open(args.raw_jsonl, encoding="utf-8")), key=lambda r: r["sent_at_s"])
    rows = [r for r in rows if r.get("ttft_ms") is not None]
    if len(rows) != len(tasks):
        raise SystemExit(f"count mismatch: {len(rows)} client records, {len(tasks)} server tasks")
    same = sum(1 for r, t in zip(rows, tasks) if r.get("prompt_tokens") == t.get("n"))
    print(f"pairs={len(rows)}  prompt length equal in {same}")

    offset = min(t["start"] - r["sent_at_s"] for r, t in zip(rows, tasks))
    split = []
    for r, t in zip(rows, tasks):
        wait = (t["start"] - r["sent_at_s"] - offset) * 1e3
        rest = r["ttft_ms"] - wait - t["prefill_ms"]
        a, b = t["prefill"]
        overlap = any(o is not t and o["prefill"][0] < b and o["prefill"][1] > a for o in tasks)
        split.append((r, wait, t["prefill_ms"], rest, overlap))

    measured = [x for x in split if x[0].get("phase") == "measured"]
    over = [x for x in measured if x[0]["ttft_ms"] > args.ttft_limit_ms]
    print(f"measured={len(measured)}  TTFT>{args.ttft_limit_ms:.0f}ms: {len(over)}")

    def median(v: list[float]) -> float:
        return statistics.median(v) if v else float("nan")

    for label, group in [("TTFT over limit", over), ("all measured", measured)]:
        print(f"  {label:16s} wait med={median([x[1] for x in group]):6.0f}  "
              f"prefill med={median([x[2] for x in group]):6.0f}  "
              f"rest med={median([x[3] for x in group]):6.0f}  "
              f"overlapped another prefill={sum(x[4] for x in group)}/{len(group)}")


if __name__ == "__main__":
    main()
