"""Run one cell on the GPU instance and write its manifest + raw-output log.

    python scripts/run_cell.py --pass-type accuracy --engine vllm \
        --base-url http://localhost:8000 --model Qwen/Qwen3-8B --quantization bf16-none \
        --engine-version 0.x.y --tool-stage 10

Writes two files to --out-dir (default runs/), named by run_id:

  <run_id>.raw.jsonl  one line per (case, trial): the tool order presented,
                      the engine's raw output, the decoded prediction, and
                      wall time. Written BEFORE the manifest, so it survives
                      a manifest-guard refusal and lets a cell be rescored
                      offline after the instance is destroyed.
  <run_id>.json       the schema-validated run manifest
                      (schema/run_manifest.schema.json).

--pass-type latency (docs/adr/0003) writes the same pair: the raw log has
one line per request (warmup, measured and cooldown alike), and the manifest
carries the percentiles and the per-class verdicts. A cell that cannot run
(engine error, never stable, prefix cache not cold) writes
<run_id>.aborted.json instead of a manifest: ADR-0003 records it as
"unsupported", which is a result, not a crash to retry.

Every guard that protects comparability lives in eval_bench/ (runner.py's
cell invariants, the vLLM concurrency guard, pipeline.py's thinking guard);
this script only wires arguments to them and never relaxes one.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from eval_bench.dataset.loader import load_cases  # noqa: E402
from eval_bench.dataset.tools import load_tools  # noqa: E402
from eval_bench.dataset.tools import to_openai_tools_subset  # noqa: E402
from eval_bench.dataset.toolsets import build_tool_set  # noqa: E402
from eval_bench.latency import (  # noqa: E402
    MIN_REQUESTS, CellAborted, run_latency_cell, tagged_query,
)
from eval_bench.pipeline import assert_no_thinking, make_predict_fn  # noqa: E402
from eval_bench.repetition import trial_order  # noqa: E402
from eval_bench.runner import (  # noqa: E402
    build_manifest, parse_vllm_diagnostics, run_accuracy_cell, validate_cell_invariants,
    write_manifest,
)

_CATALOGS = {
    10: _ROOT / "schema" / "tools.json",            # Article-1 overlay point: original 40 only
    50: _ROOT / "schema" / "tools_extended.json",
    100: _ROOT / "schema" / "tools_extended.json",
}

# docs/engine-performance-plan.md: Qwen3 non-thinking recommended sampling.
_QWEN3_NON_THINKING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pass-type", required=True, choices=["accuracy", "accuracy_under_load", "latency"])
    p.add_argument("--engine", required=True, choices=["vllm", "llama_cpp", "transformers_hf"])
    p.add_argument("--base-url", help="vllm / llama_cpp server URL")
    p.add_argument("--model", required=True)
    p.add_argument("--quantization", required=True, help="e.g. bf16-none, gguf-bf16, awq-4bit")
    p.add_argument("--engine-version", required=True,
                   help="llama_cpp: the build's git commit hash (enforced)")
    p.add_argument("--tool-stage", type=int, required=True, choices=sorted(_CATALOGS))
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--n", type=int, default=5, help="accuracy passes: trials per case")
    p.add_argument("--n-requests", type=int,
                   help=f"latency: measured requests, fixed before the run (>= {MIN_REQUESTS} "
                        "unless --run-id-prefix is smoke-/pilot-)")
    p.add_argument("--request-timeout", type=float, default=120.0,
                   help="latency: a request past this counts as an exceedance")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--temperature", type=float, default=_QWEN3_NON_THINKING["temperature"],
                   help="the third pass uses 0 (greedy), docs/adr/0002")
    p.add_argument("--top-p", type=float, default=_QWEN3_NON_THINKING["top_p"])
    p.add_argument("--top-k", type=int, default=_QWEN3_NON_THINKING["top_k"])
    p.add_argument("--min-p", type=float, default=_QWEN3_NON_THINKING["min_p"])
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vllm-startup-log", type=pathlib.Path,
                   help="required for vllm at concurrency > 1 (the queueing guard)")
    p.add_argument("--run-id-prefix", default="", help="smoke- / pilot- / empty for main runs")
    p.add_argument("--max-cases", type=int,
                   help="smoke- / pilot- only: run the first N cases (by id); a main run uses them all")
    p.add_argument("--data-dir", type=pathlib.Path, default=_ROOT / "data")
    p.add_argument("--out-dir", type=pathlib.Path, default=_ROOT / "runs")
    p.add_argument("--instance-type", default="g6.xlarge")
    p.add_argument("--region", default="ap-northeast-1")
    args = p.parse_args(argv)

    if args.engine in ("vllm", "llama_cpp") and not args.base_url:
        p.error(f"--base-url is required for --engine {args.engine}")
    if args.pass_type == "accuracy" and args.concurrency != 1:
        p.error("plain accuracy is concurrency=1 by definition; use --pass-type accuracy_under_load")
    if args.engine == "vllm" and args.concurrency > 1 and args.vllm_startup_log is None:
        p.error("--vllm-startup-log is required for vllm at concurrency > 1")
    trial_run = args.run_id_prefix.startswith(("smoke-", "pilot-"))
    if args.max_cases is not None and not trial_run:
        p.error("--max-cases is for smoke-/pilot- runs only: the case set is fixed for main runs (docs/adr/0004)")
    if args.pass_type == "latency":
        if args.n_requests is None:
            p.error("--n-requests is required for --pass-type latency")
        if args.n_requests < MIN_REQUESTS and not trial_run:
            p.error(f"--n-requests must be >= {MIN_REQUESTS} for a main run (docs/adr/0003)")
        if args.engine == "transformers_hf" and args.concurrency != 1:
            p.error("transformers_hf is the concurrency=1 control point only")
    return args


def _build_backend(args: argparse.Namespace, sampling: dict[str, Any]):
    if args.engine == "transformers_hf":
        from eval_bench.adapters.transformers_hf import TransformersHFAdapter
        from eval_bench.decoders.base import HermesTagDecoder
        return TransformersHFAdapter.from_pretrained(args.model, sampling), HermesTagDecoder()

    from eval_bench.adapters.openai_compatible import LlamaCppAdapter, VLLMAdapter
    from eval_bench.decoders.base import OpenAIToolCallDecoder
    cls = VLLMAdapter if args.engine == "vllm" else LlamaCppAdapter
    return cls(base_url=args.base_url, model=args.model, sampling_params=sampling), OpenAIToolCallDecoder()


def _query_gpu() -> tuple[str, float]:
    """GPU name and memory from nvidia-smi, so the manifest records what the
    run actually had rather than what the plan assumed."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
    name, mib = (s.strip() for s in out.split(","))
    return name, round(float(mib) / 1024, 1)


def _no_semantic_judge(expected: str, predicted: str) -> bool:
    raise RuntimeError(
        "a semantic-scored field was scored, but schema/tools.json 2.1.1 has none -- "
        "the catalog or a case changed without choosing a judge"
    )


class _RawLog:
    """Thread-safe collector for pipeline.make_predict_fn's recorder hook.

    The recorder sees (case, tool order) but not the trial index; the index
    is recovered by recomputing each trial's order with the same
    repetition.trial_order the run itself used.
    """

    def __init__(self, cases, tools, stage, n, seed_base):
        self._lock = threading.Lock()
        self.rows: list[dict[str, Any]] = []
        self._trial_of: dict[tuple[str, tuple[str, ...]], list[int]] = {}
        for case in cases:
            tool_set = build_tool_set(case, tools, stage, seed=0)
            for trial in range(n):
                key = (case["id"], tuple(trial_order(case["id"], tool_set, stage, trial, seed_base)))
                self._trial_of.setdefault(key, []).append(trial)

    def __call__(self, case, tool_names, raw, pred, elapsed):
        trials = self._trial_of.get((case["id"], tuple(tool_names)), [])
        row = {
            "case_id": case["id"],
            "trial": trials[0] if len(trials) == 1 else None,  # None only if two trials drew the same order
            "tool_order": list(tool_names),
            "raw_output": raw,
            "decoded": {"tool": pred.tool, "args": pred.args},
            "elapsed_s": round(elapsed, 4),
        }
        with self._lock:
            self.rows.append(row)

    def write(self, path: pathlib.Path) -> None:
        _write_jsonl(path, sorted(self.rows, key=lambda r: (r["case_id"], r["trial"] if r["trial"] is not None else -1)))


def _reset_vllm_prefix_cache(base_url: str, attempts: int = 3) -> None:
    """Empty vLLM's prefix cache at the cell boundary (docs/adr/0003). The
    endpoint exists only with VLLM_SERVER_DEV_MODE=1 and answers
    {"success": bool}; it fails while blocks are still held, so retry."""
    import requests
    for _ in range(attempts):
        response = requests.post(f"{base_url.rstrip('/')}/reset_prefix_cache", timeout=30)
        if response.status_code == 404:
            raise RuntimeError("POST /reset_prefix_cache returned 404: start vLLM with VLLM_SERVER_DEV_MODE=1")
        response.raise_for_status()
        if response.json().get("success"):
            return
        time.sleep(1)
    raise RuntimeError(f"vLLM's prefix cache reset did not succeed in {attempts} attempts")


def _make_sender(adapter, cases, tools, stage: int, timeout: float, run_tag: str):
    """Each case keeps one tool set in one order for the whole cell (its
    stage subset in build_tool_set's seed-0 order): fixed tool definitions,
    a different query every time, as in production (docs/adr/0003 4)."""
    tool_sets = {c["id"]: to_openai_tools_subset(tools, build_tool_set(c, tools, stage, seed=0))
                 for c in cases}

    def send(case, index):
        result = adapter.generate_timed(tagged_query(case["query"], run_tag, index),
                                        tool_sets[case["id"]], timeout)
        if result.status == "ok":
            assert_no_thinking(json.dumps({"content": result.content,
                                           "reasoning_content": result.reasoning}))
        return result
    return send


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _finish(args, run_id: str, cell: dict[str, Any], results: dict[str, Any],
            repetition: dict[str, Any] | None) -> pathlib.Path:
    if args.engine == "vllm" and args.concurrency > 1:
        results["vllm_diagnostics"] = parse_vllm_diagnostics(
            args.vllm_startup_log.read_text(encoding="utf-8", errors="replace"))
    gpu, vram_gb = _query_gpu()
    hardware = {"instance_type": args.instance_type, "gpu": gpu, "vram_gb": vram_gb,
                "region": args.region}
    manifest = build_manifest(run_id, args.pass_type, cell, hardware, results, repetition=repetition)
    manifest_path = write_manifest(manifest, args.out_dir)
    print(f"wrote {manifest_path}")
    return manifest_path


def _run_latency(args, run_id: str, cell: dict[str, Any], cases, tools, adapter) -> int:
    if args.engine == "vllm":
        _reset_vllm_prefix_cache(args.base_url)
    send = _make_sender(adapter, cases, tools, args.tool_stage, args.request_timeout,
                        run_tag=secrets.token_hex(4))
    raw_path = args.out_dir / f"{run_id}.raw.jsonl"
    try:
        rows, results = run_latency_cell(cases, send, args.concurrency, args.n_requests)
    except CellAborted as exc:
        _write_jsonl(raw_path, getattr(exc, "rows", []))
        aborted_path = args.out_dir / f"{run_id}.aborted.json"
        aborted_path.write_text(json.dumps({
            "run_id": run_id, "pass_type": args.pass_type, "cell": cell,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "outcome": "unsupported", "reason": str(exc),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"cell aborted: {exc}\nwrote {raw_path}\nwrote {aborted_path}")
        return 2
    _write_jsonl(raw_path, rows)
    print(f"wrote {raw_path}")
    _finish(args, run_id, cell, results, repetition=None)
    print(json.dumps({name: (v["verdict"], v["exceeded"], v["n"]) for name, v in results["slo"].items()},
                     ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sampling = {
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "min_p": args.min_p, "max_tokens": args.max_tokens, "seed": args.seed,
    }
    catalog_path = _CATALOGS[args.tool_stage]
    tools = load_tools(catalog_path)
    catalog_version = json.loads(catalog_path.read_text(encoding="utf-8"))["version"]
    cases = load_cases(args.data_dir)
    if args.max_cases is not None:
        cases = sorted(cases, key=lambda c: c["id"])[:args.max_cases]

    cell: dict[str, Any] = {
        "engine": args.engine,
        "engine_version": args.engine_version,
        "model": args.model,
        "quantization": args.quantization,
        "sampling_params": sampling,
        "prompt_layout": "stable_first",
        "tool_catalog_version": catalog_version,
        "tool_stage": args.tool_stage,
    }
    if args.pass_type != "accuracy":  # the schema forbids stating concurrency on plain accuracy
        cell["concurrency"] = args.concurrency

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = (f"{args.run_id_prefix}{args.pass_type}-{args.engine}-{args.quantization}"
              f"-s{args.tool_stage}-c{args.concurrency}-t{args.temperature:g}-b{args.seed_base}-{stamp}")

    adapter, decoder = _build_backend(args, sampling)
    if args.pass_type == "latency":
        validate_cell_invariants(args.pass_type, cell, tools)
        return _run_latency(args, run_id, cell, cases, tools, adapter)

    raw_log = _RawLog(cases, tools, args.tool_stage, args.n, args.seed_base)
    predict = make_predict_fn(adapter, decoder, tools, recorder=raw_log)

    results = run_accuracy_cell(cases, tools, cell, args.pass_type, predict,
                                _no_semantic_judge, n=args.n, seed_base=args.seed_base)
    raw_path = args.out_dir / f"{run_id}.raw.jsonl"
    raw_log.write(raw_path)
    print(f"wrote {raw_path}")
    _finish(args, run_id, cell, results, repetition={"n": args.n, "seed_base": args.seed_base})
    print(json.dumps({k: results[k] for k in ("selection_mean", "selection_ci95", "consistency")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
