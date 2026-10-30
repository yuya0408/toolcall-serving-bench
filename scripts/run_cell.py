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
    MAIN_N_REQUESTS, CellAborted, new_run_tag, run_latency_cell, tagged_query,
)
from eval_bench.pipeline import assert_no_thinking, make_predict_fn  # noqa: E402
from eval_bench.repetition import trial_order  # noqa: E402
from eval_bench import capacity  # noqa: E402
from eval_bench.runner import (  # noqa: E402
    build_manifest, run_accuracy_cell, validate_cell_invariants, write_manifest,
)
from eval_bench.scorers.separated import score_case  # noqa: E402

_CATALOGS = {
    10: _ROOT / "schema" / "tools.json",            # Article-1 overlay point: original 40 only
    50: _ROOT / "schema" / "tools_extended.json",
    100: _ROOT / "schema" / "tools_extended.json",
}

# Engines run as a server, reached over the OpenAI-compatible API.
SERVED_ENGINES = ("vllm", "sglang", "tensorrt_llm", "llama_cpp")

# the engine-performance plan (kept outside this repo): Qwen3 non-thinking recommended sampling.
_QWEN3_NON_THINKING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pass-type", required=True, choices=["accuracy", "accuracy_under_load", "latency"])
    p.add_argument("--engine", required=True, choices=["vllm", "sglang", "tensorrt_llm", "llama_cpp", "transformers_hf"])
    p.add_argument("--base-url", help="the served engine's URL (every engine but transformers_hf)")
    p.add_argument("--model", required=True)
    p.add_argument("--quantization", required=True, help="e.g. bf16-none, gguf-bf16, awq-4bit")
    p.add_argument("--engine-version", required=True,
                   help="llama_cpp: the build's git commit hash (enforced)")
    p.add_argument("--tool-stage", type=int, required=True, choices=sorted(_CATALOGS))
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--n", type=int, default=5, help="accuracy passes: trials per case")
    p.add_argument("--n-requests", type=int,
                   help=f"latency: measured requests, fixed before the run ({MAIN_N_REQUESTS} "
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
    p.add_argument("--capacity-check", type=pathlib.Path,
                   help="the pre-check scripts/gpu/run_cells.py wrote; required for a served engine "
                        "at concurrency > 1 (docs/adr/0005)")
    p.add_argument("--run-id-prefix", default="", help="smoke- / pilot- / empty for main runs")
    p.add_argument("--max-cases", type=int,
                   help="smoke- / pilot- only: run the first N cases (by id); a main run uses them all")
    p.add_argument("--data-dir", type=pathlib.Path, default=_ROOT / "data")
    p.add_argument("--out-dir", type=pathlib.Path, default=_ROOT / "runs")
    p.add_argument("--instance-type", default="g6.xlarge")
    p.add_argument("--region", default="ap-northeast-1")
    args = p.parse_args(argv)

    if args.engine in SERVED_ENGINES and not args.base_url:
        p.error(f"--base-url is required for --engine {args.engine}")
    if args.pass_type == "accuracy" and args.concurrency != 1:
        p.error("plain accuracy is concurrency=1 by definition; use --pass-type accuracy_under_load")
    if args.engine in SERVED_ENGINES and args.concurrency > 1 and args.capacity_check is None:
        p.error("--capacity-check is required for a served engine at concurrency > 1 (docs/adr/0005)")
    trial_run = args.run_id_prefix.startswith(("smoke-", "pilot-"))
    if args.max_cases is not None and not trial_run:
        p.error("--max-cases is for smoke-/pilot- runs only: the case set is fixed for main runs (docs/adr/0004)")
    if args.pass_type == "latency":
        if args.n_requests is None:
            p.error("--n-requests is required for --pass-type latency")
        if args.n_requests != MAIN_N_REQUESTS and not trial_run:
            p.error(f"--n-requests must be {MAIN_N_REQUESTS} for a main run (docs/adr/0003, 2026-10-23)")
        if args.engine == "transformers_hf" and args.concurrency != 1:
            p.error("transformers_hf is the concurrency=1 control point only")
    return args


def _build_backend(args: argparse.Namespace, sampling: dict[str, Any]):
    if args.engine == "transformers_hf":
        from eval_bench.adapters.transformers_hf import TransformersHFAdapter
        from eval_bench.decoders.base import HermesTagDecoder
        return TransformersHFAdapter.from_pretrained(args.model, sampling), HermesTagDecoder()

    from eval_bench.adapters import openai_compatible as oc
    from eval_bench.decoders.base import OpenAIToolCallDecoder
    cls = {"vllm": oc.VLLMAdapter, "sglang": oc.SGLangAdapter, "tensorrt_llm": oc.TensorRTLLMAdapter,
           "llama_cpp": oc.LlamaCppAdapter}[args.engine]
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


def _flush_sglang_cache(base_url: str) -> None:
    """Empty SGLang's radix cache at the cell boundary, as for vLLM (the
    server is also fresh). /flush_cache refuses while requests are running."""
    import requests
    response = requests.post(f"{base_url.rstrip('/')}/flush_cache", timeout=30)
    response.raise_for_status()


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


class _CapacityWatch:
    """The post-check's bookkeeping around one cell (docs/adr/0005 3): vLLM's
    and SGLang's cumulative /metrics counters read before and after; for
    TensorRT-LLM, its iteration statistics polled during the cell (each read of
    /metrics drains them, so they cannot be read once at the end; docs/adr/0009 2);
    nothing to read for llama.cpp, whose check comes from the raw output alone."""

    POLL_S = 2.0

    def __init__(self, args):
        self.args = args
        self.pre = None
        self.before = None
        self._stats: list[dict[str, Any]] = []
        self._stop = None
        self._thread = None
        if args.capacity_check is not None:
            self.pre = json.loads(args.capacity_check.read_text(encoding="utf-8"))
            expected = capacity.per_request_max(self.pre["max_prompt_tokens_planned"], args.max_tokens)
            if (self.pre["required"]["per_request_tokens"] != expected
                    or self.pre["required"]["concurrency"] != args.concurrency):
                raise SystemExit("--capacity-check was computed for another max_tokens or concurrency")

    def start(self) -> None:
        if self.pre is not None and self.args.engine in capacity.PREEMPTION_COUNTERS:
            counters, _ = capacity.PREEMPTION_COUNTERS[self.args.engine]
            self.before = counters(capacity.fetch_metrics(self.args.base_url))
        if self.pre is not None and self.args.engine == "tensorrt_llm":
            import threading
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

    def _poll(self) -> None:
        while True:
            try:
                self._stats.extend(capacity.fetch_json(self.args.base_url, "/metrics"))
            except Exception:   # a missed poll only shortens the record; the check reports it
                pass
            if self._stop.wait(self.POLL_S):
                return

    def result(self, prompt_tokens, length_finished: int,
               output_limit_finished: int) -> dict[str, Any] | None:
        if self.pre is None:
            return None
        delta = None
        if self.args.engine in capacity.PREEMPTION_COUNTERS:
            counters, difference = capacity.PREEMPTION_COUNTERS[self.args.engine]
            delta = difference(self.before, counters(capacity.fetch_metrics(self.args.base_url)))
        active = None
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            active = capacity.trtllm_active(self._stats)
        post = capacity.post_check(self.args.engine, self.pre["max_prompt_tokens_planned"],
                                   prompt_tokens, vllm_delta=delta, length_finished=length_finished,
                                   output_limit_finished=output_limit_finished, trt_active=active,
                                   concurrency=self.args.concurrency)
        return {**self.pre, "post_check": post}


def _finish(args, run_id: str, cell: dict[str, Any], results: dict[str, Any],
            repetition: dict[str, Any] | None) -> pathlib.Path:
    gpu, vram_gb = _query_gpu()
    hardware = {"instance_type": args.instance_type, "gpu": gpu, "vram_gb": vram_gb,
                "region": args.region}
    manifest = build_manifest(run_id, args.pass_type, cell, hardware, results, repetition=repetition)
    manifest_path = write_manifest(manifest, args.out_dir)
    print(f"wrote {manifest_path}")
    return manifest_path


def _reference_accuracy(rows, cases, decoder, engine: str) -> dict[str, Any]:
    """The measured requests' tool calls scored like the accuracy pass, as a
    reference only (docs/adr/0003 J): sampled, with a request id appended to
    each query, so never a verdict."""
    by_id = {c["id"]: c for c in cases}
    scores = []
    for row in rows:
        if row["phase"] != "measured" or row["status"] != "ok":
            continue
        if engine == "transformers_hf":
            raw = row["content"]
        else:
            raw = json.dumps({"tool_calls": [{"function": tc} for tc in row.get("tool_calls") or []]})
        scores.append(score_case(by_id[row["case_id"]], decoder.decode(raw, {}), _no_semantic_judge))

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    exact = [float(sc.arg_exact) for sc in scores if sc.arg_exact is not None]
    return {"n": len(scores), "selection_mean": mean([float(sc.selection) for sc in scores]),
            "arg_exact_mean": mean(exact)}


def _run_latency(args, run_id: str, cell: dict[str, Any], cases, tools, adapter, decoder) -> int:
    if args.engine == "vllm":
        _reset_vllm_prefix_cache(args.base_url)
    elif args.engine == "sglang":
        _flush_sglang_cache(args.base_url)
    watch = _CapacityWatch(args)
    watch.start()
    send = _make_sender(adapter, cases, tools, args.tool_stage, args.request_timeout,
                        run_tag=new_run_tag())
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
    results["reference_accuracy"] = _reference_accuracy(rows, cases, decoder, args.engine)
    check = watch.result([r["prompt_tokens"] for r in rows],
                         *_length_stops([(r.get("finish_reason"), r.get("output_tokens")) for r in rows],
                                        args.max_tokens))
    if check is not None:
        results["capacity_check"] = check
    _finish(args, run_id, cell, results, repetition=None)
    print(json.dumps({name: (v["verdict"], v["exceeded"], v["n"]) for name, v in results["slo"].items()},
                     ensure_ascii=False))
    return _post_check_exit(check)


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
        return _run_latency(args, run_id, cell, cases, tools, adapter, decoder)

    raw_log = _RawLog(cases, tools, args.tool_stage, args.n, args.seed_base)
    predict = make_predict_fn(adapter, decoder, tools, recorder=raw_log)

    watch = _CapacityWatch(args)
    watch.start()
    try:
        results = run_accuracy_cell(cases, tools, cell, args.pass_type, predict,
                                    _no_semantic_judge, n=args.n, seed_base=args.seed_base)
    except BaseException:
        # Keep what was answered, for the diagnosis; a name the offline scripts
        # do not read (*.raw.jsonl), since the cell did not finish.
        raw_log.write(args.out_dir / f"{run_id}.raw.partial.jsonl")
        raise
    raw_path = args.out_dir / f"{run_id}.raw.jsonl"
    raw_log.write(raw_path)
    print(f"wrote {raw_path}")
    prompt_tokens, finishes = _usage_of(raw_log.rows)
    check = watch.result(prompt_tokens, *_length_stops(finishes, args.max_tokens))
    if check is not None:
        results["capacity_check"] = check
    _finish(args, run_id, cell, results, repetition={"n": args.n, "seed_base": args.seed_base})
    print(json.dumps({k: results[k] for k in ("selection_mean", "selection_ci95", "consistency")},
                     ensure_ascii=False))
    return _post_check_exit(check)


def _usage_of(rows) -> tuple[list, list]:
    """Prompt lengths and (finish_reason, completion tokens) from the accuracy
    passes' raw outputs (the reserved keys the OpenAI-compatible adapter adds)."""
    from eval_bench.adapters.openai_compatible import FINISH_REASON_KEY, USAGE_KEY
    prompt_tokens, finishes = [], []
    for row in rows:
        try:
            message = json.loads(row["raw_output"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(message, dict):
            continue
        usage = message.get(USAGE_KEY) or {}
        prompt_tokens.append(usage.get("prompt_tokens"))
        finishes.append((message.get(FINISH_REASON_KEY), usage.get("completion_tokens")))
    return prompt_tokens, finishes


def _length_stops(finishes, max_tokens: int) -> tuple[int, int]:
    """Responses that stopped at the length limit, and those of them that had
    generated max_tokens (their own output limit, not the context). One whose
    output length is unknown is not counted as the output limit."""
    lengths = [n for reason, n in finishes if reason == "length"]
    return len(lengths), sum(1 for n in lengths if n is not None and n >= max_tokens)


def _post_check_exit(check) -> int:
    """3 = the manifest was written but the capacity post-check failed: the
    cell's numbers are not used for a verdict (docs/adr/0005 3)."""
    if check is not None and not check["post_check"]["ok"]:
        print(f"capacity post-check failed: {check['post_check']['problems']}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
