"""Read the vLLM FP8 check (docs/vllm-fp8-tensor-check.md), offline from the
collected folder and phase 2's.

    python scripts/fp8_tensor_check_summary.py runs/<this check> \
        runs/gpu-20261028-i-0bd6a9b5da22d5bf5

Speed (T1, B1, C, T2): (1) the TPOT rise per overlapping prefill (the median
TPOT with 3 other prefills inside the generation minus with 0, over 3;
scripts/tpot_prefill_overlap.py), (2) the TPOT and (3) the TTFT median, each
against the mean of T1 and T2 -- the value to read -- and against T1 and T2
alone, with the cell's mean SM clock. No line sorts a difference into "same"
or "different"; the document reads them by size and direction. Phase 2's
TensorRT-LLM FP8 and vLLM FP8 at concurrency 4 are printed beside them. Each
cell also shows the capacity post-check (docs/adr/0005), the linear kernel
vLLM selected, the quantization its /server_info reports, and how many
requests' prompt_tokens differ from phase 2's vLLM FP8 at the same index.

Accuracy (B2): first phase 2's gate is recomputed from its folder and must
give vLLM FP8 -2.08pt and TensorRT-LLM FP8 -1.43pt; then B2 is judged against
the bf16s that passed step 1 (vLLM and TensorRT-LLM; docs/adr/0009 5), by the
same pooled_noninferiority. vLLM bf16 alone as the reference, and B2's
disagreements with each reference, are printed for reference.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import pathlib
import statistics
import sys
from datetime import datetime, timezone
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import tpot_prefill_overlap as tpo  # noqa: E402

T1, B1, C, T2 = "lat-trt-fp8-c4-r1", "lat-vllm-fp8tensor-c4", "lat-vllm-fp8modelopt-c4", "lat-trt-fp8-c4-r2"
B2 = "ni-vllm-fp8tensor-c1"
PHASE2 = {"TRT-LLM FP8": "lat-trt-fp8-c4", "vLLM FP8": "lat-vllm-fp8-c4"}
PROMPT_REFERENCE = "lat-vllm-fp8-c4"            # phase 2's vLLM: same request sequence, same tokenizer
PHASE2_FP8_BOUNDS = {"vllm-fp8": -2.08, "trt-fp8": -1.43}   # docs/gpu-main-batching-2026-10-30.md
KERNELS = ("CutlassFP8ScaledMMLinearKernel", "MarlinFP8ScaledMMLinearKernel", "TritonFp8BlockScaledMMKernel")
FUSED_SCALE_WARNING = "static input scale differs"
OVERLAP = 3   # most requests at concurrency 4 overlap about 3 prefills (docs/analysis-batching.md)


def rise_per_overlap(pairs: list[tuple[int, float]]) -> float | None:
    """(median TPOT at OVERLAP overlaps - median at 0) / OVERLAP, or None if a group is empty."""
    zero = [t for n, t in pairs if n == 0]
    many = [t for n, t in pairs if n == OVERLAP]
    if not zero or not many:
        return None
    return (statistics.median(many) - statistics.median(zero)) / OVERLAP


def _one(folder: pathlib.Path, pattern: str) -> pathlib.Path:
    files = sorted(folder.glob(pattern))
    if len(files) != 1:
        raise SystemExit(f"{folder}: expected one {pattern}, found {len(files)}")
    return files[0]


def _utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def mean_clock(gpu_csv: pathlib.Path, start: datetime, end: datetime) -> float | None:
    """Mean SM clock (MHz) of the gpu.csv rows inside [start, end]; the instance logs in UTC."""
    if not gpu_csv.exists():
        return None
    with gpu_csv.open(encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    col = next((i for i, h in enumerate(header) if h.startswith("clocks.current.sm") or h.startswith("clocks.sm")), None)
    if col is None:
        return None
    clocks = []
    for row in rows[1:]:
        try:
            t = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            mhz = float(row[col].strip().split()[0])
        except (ValueError, IndexError):
            continue
        if start <= t <= end:
            clocks.append(mhz)
    return statistics.mean(clocks) if clocks else None


def find_key(obj: Any, key: str) -> Any:
    """The first value under key anywhere in a JSON tree (vLLM's /server_info nests its config)."""
    if isinstance(obj, dict):
        if obj.get(key) is not None:
            return obj[key]
        children = obj.values()
    elif isinstance(obj, list):
        children = obj
    else:
        return None
    for child in children:
        found = find_key(child, key)
        if found is not None:
            return found
    return None


def server_facts(cell_dir: pathlib.Path) -> dict[str, Any]:
    log = (cell_dir / "server.log").read_text(errors="replace") if (cell_dir / "server.log").exists() else ""
    info = cell_dir / "engine-server_info-before.json"
    quantization = find_key(json.loads(info.read_text(errors="replace")), "quantization") if info.exists() else None
    return {"kernels": [k for k in KERNELS if k in log], "fused_scale_warning": FUSED_SCALE_WARNING in log,
            "quantization": quantization}


def cell_speed(run_dir: pathlib.Path, name: str, status: dict[str, Any] | None = None) -> dict[str, Any]:
    folder = run_dir / name
    rows = tpo.load(str(_one(folder, "*.raw.jsonl")))
    manifest = json.loads(_one(folder, "latency-*Z.json").read_text(encoding="utf-8"))
    results = manifest["results"]
    post = results.get("capacity_check", {}).get("post_check", {})
    out = {"rise_ms": rise_per_overlap(tpo.overlaps(rows)), "tpot_ms": results["tpot_ms"]["p50"],
           "ttft_ms": results["ttft_ms"]["p50"], "post_check_ok": post.get("ok"), "rows": rows,
           **server_facts(folder)}
    if status and "run_started_at" in status and "finished_at" in status:
        out["sm_mhz"] = mean_clock(run_dir / "gpu.csv", _utc(status["run_started_at"]), _utc(status["finished_at"]))
    return out


def pct(x: float | None, ref: float | None) -> str:
    return "-" if x is None or not ref else f"{100 * (x / ref - 1):+.1f}%"


def prompt_differences(rows: list[dict], reference: list[dict]) -> tuple[int, int]:
    """(requests whose prompt_tokens differ from the reference's at the same index, largest difference)."""
    ref = {r["index"]: r.get("prompt_tokens") for r in reference}
    diffs = [abs(r["prompt_tokens"] - ref[r["index"]]) for r in rows
             if r.get("prompt_tokens") is not None and ref.get(r["index"]) is not None]
    differing = [d for d in diffs if d]
    return len(differing), max(differing, default=0)


def speed(run_dir: pathlib.Path, phase2: pathlib.Path) -> None:
    statuses = {}
    status_path = run_dir / "status.jsonl"
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                statuses[row["name"]] = row
    cells = {}
    for name in (T1, B1, C, T2):
        if statuses.get(name, {}).get("outcome") not in (None, "ok"):
            print(f"{name}: {statuses[name]['outcome']} -- not read")
            continue
        cells[name] = cell_speed(run_dir, name, statuses.get(name))
    reference = tpo.load(str(_one(phase2 / PROMPT_REFERENCE, "*.raw.jsonl")))
    print("| cell | metric | value | vs mean(T1,T2) | vs T1 | vs T2 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for metric in ("rise_ms", "tpot_ms", "ttft_ms"):
        both = [cells[t][metric] for t in (T1, T2) if t in cells and cells[t][metric] is not None]
        mean = statistics.mean(both) if len(both) == 2 else None
        for name, c in cells.items():
            v = c[metric]
            t1 = cells.get(T1, {}).get(metric)
            t2 = cells.get(T2, {}).get(metric)
            print(f"| {name} | {metric} | {'-' if v is None else f'{v:.2f}'} | {pct(v, mean)} | {pct(v, t1)} | {pct(v, t2)} |")
    print()
    print("| cell | mean SM clock | post-check | kernels | quantization | fused-scale warning | prompt_tokens differing (max) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, c in cells.items():
        n, worst = prompt_differences(c["rows"], reference)
        mhz = c.get("sm_mhz")
        print(f"| {name} | {'-' if mhz is None else f'{mhz:.0f} MHz'} | {c['post_check_ok']} | "
              f"{', '.join(c['kernels']) or '-'} | {c['quantization']} | {c['fused_scale_warning']} | {n} ({worst}) |")
    print()
    print("phase 2 (another instance, for reference):")
    for label, name in PHASE2.items():
        c = cell_speed(phase2, name)
        print(f"  {label}: rise {c['rise_ms']:.2f} ms, TPOT {c['tpot_ms']:.1f} ms, TTFT {c['ttft_ms']:.0f} ms")


def accuracy(run_dir: pathlib.Path, phase2: pathlib.Path) -> None:
    from eval_bench.dataset.loader import load_cases
    from eval_bench.flips import load_arm
    from eval_bench.noninferiority import _selection, batching_gate, load_batching_arms, pooled_noninferiority
    cases = {c["id"]: c for c in load_cases(_ROOT / "data")}
    arms = load_batching_arms([phase2], cases)
    gate = batching_gate(arms)
    for name, want in PHASE2_FP8_BOUNDS.items():
        got = 100 * gate["quantized"][name]["lower_bound"]   # a fraction; the record is in pt
        print(f"phase 2 {name}: lower bound {got:+.2f}pt (recorded {want:+.2f})")
        if round(got, 2) != want:
            raise SystemExit("phase 2's gate did not reproduce; B2 is not judged")
    if gate["passed_bf16"] != ["vllm-bf16", "trt-bf16"]:
        raise SystemExit(f"phase 2's step 1 passed {gate['passed_bf16']}, not vLLM and TensorRT-LLM bf16")
    folder = run_dir / B2
    b2 = _selection(load_arm(_one(folder, "*.raw.jsonl")), cases)
    refs = [arms[n] for n in gate["passed_bf16"]]
    result = pooled_noninferiority(refs, b2)
    print(f"B2 against {gate['passed_bf16']}: selection diff {100 * result['selection_diff']:+.2f}pt, "
          f"lower bound {100 * result['lower_bound']:+.2f}pt -> {result['verdict']}")
    alone = pooled_noninferiority([arms["vllm-bf16"]], b2)
    print(f"  for reference, against vLLM bf16 alone: lower bound {100 * alone['lower_bound']:+.2f}pt")
    for n in gate["passed_bf16"]:
        flips = sum(b2[k] != arms[n][k] for k in b2)
        print(f"  disagreements with {n}: {flips} of {len(b2)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("phase2_dir", type=pathlib.Path)
    args = ap.parse_args(argv)
    speed(args.run_dir, args.phase2_dir)
    print()
    if (args.run_dir / B2).exists():
        accuracy(args.run_dir, args.phase2_dir)
    else:
        print(f"{B2}: not collected -- accuracy not judged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
