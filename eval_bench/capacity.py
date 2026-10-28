"""Whether a concurrency>1 cell actually ran concurrently (docs/adr/0005).

A cell at concurrency c is only a measurement of "c requests at once" if the
engine's KV cache holds c requests even when every one of them grows to the
longest it can be (the guarantee-style definition, ADR-0005 2). This module
does the arithmetic and reads the engines' machine-facing endpoints:

- the pre-check, before the cell runs: c x (the longest request, rounded up
  to the engine's allocation unit) <= the KV capacity the server reports
  (vLLM: /metrics `vllm:cache_config_info`; SGLang: /metrics
  `sglang:max_total_num_tokens`; TensorRT-LLM: /metrics kvCacheStats;
  llama.cpp: the --ctx-size it was started with, and /props `n_ctx` as the
  per-slot limit);
- the post-check, after: vLLM preempted nothing (the `vllm:num_preemptions`
  counter's delta), SGLang retracted nothing (`sglang:num_retracted_requests_total`);
  llama.cpp and TensorRT-LLM, which have no such counter, returned no
  response cut off by the length limit;
- the longest request: the longest prompt, counted offline by
  scripts/count_prompt_tokens.py into schema/prompt_budget.json, plus
  max_tokens.

Pure functions plus two small HTTP readers; scripts/gpu/run_cells.py and
scripts/run_cell.py do the orchestration.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from typing import Any, Iterable, Optional

_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT_BUDGET_PATH = _ROOT / "schema" / "prompt_budget.json"

# One allocation unit on top of the counted length, for tokenizer differences
# between engines that were not checked (ADR-0005 2). The offline count matched
# vLLM's prompt_tokens, and llama.cpp renders the same tokens as HF; the 4 tokens
# llama.cpp seemed to lack in the 2026-10-18 smoke were the random hex request
# id, not the engine (2026-10-20).
MARGIN_UNITS = 1


# ---------------------------------------------------------------- the budget

def cases_digest(cases: Iterable[dict[str, Any]]) -> str:
    """A digest of what the prompts are built from, so a stale budget (cases
    edited after the count) is caught instead of silently trusted."""
    canonical = json.dumps(sorted(cases, key=lambda c: c["id"]), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_prompt_budget(stage: int, catalog_version: str, cases: list[dict[str, Any]],
                       path: pathlib.Path = PROMPT_BUDGET_PATH) -> dict[str, Any]:
    """The counted longest prompt for this stage; refuses one counted against
    a different catalog or case set."""
    budgets = json.loads(path.read_text(encoding="utf-8"))
    entry = budgets.get(str(stage))
    if entry is None:
        raise ValueError(f"{path.name} has no entry for tool_stage {stage}: "
                         f"run scripts/count_prompt_tokens.py --tool-stage {stage} --write-budget")
    if entry["tool_catalog_version"] != catalog_version or entry["cases_sha256"] != cases_digest(cases):
        raise ValueError(f"{path.name} (tool_stage {stage}) was counted against another catalog or case set: "
                         f"recount with scripts/count_prompt_tokens.py --tool-stage {stage} --write-budget")
    return entry


def max_prompt_for(budget: dict[str, Any], engine: str) -> int:
    """The longest prompt this engine will see: its own count where it renders
    the tools differently (SGLang, scripts/count_prompt_tokens.py --engine),
    the shared one otherwise."""
    return budget.get("by_engine", {}).get(engine, {}).get("max_prompt_tokens", budget["max_prompt_tokens"])


def per_request_max(max_prompt_tokens: int, max_tokens: int) -> int:
    return max_prompt_tokens + max_tokens


def required_tokens(per_request: int, concurrency: int, unit_tokens: int,
                    margin_units: int = MARGIN_UNITS) -> int:
    """c x the longest request, each rounded up to whole allocation units."""
    units = math.ceil((per_request + margin_units * unit_tokens) / unit_tokens)
    return units * unit_tokens * concurrency


# llama.cpp's KV layouts in the order a cell tries them (ADR-0005, 2026-10-23):
# one cache per slot (the default with a fixed --parallel), then one cache shared
# by every slot if the per-slot padding no longer fits in memory.
LLAMA_KV_LAYOUTS = ("split", "unified")


def llama_ctx_size(per_request: int, concurrency: int, kv: str = "split", round_to: int = 256) -> int:
    """llama.cpp's --ctx-size for a cell: the requirement (ADR-0005 1), rounded
    up the way the server pads it, so what we pass is what it allocates. A
    split cache pads every slot to round_to tokens (src/llama-context.cpp); a
    unified one pads the total once."""
    if kv == "split":
        per_slot = required_tokens(per_request, 1, unit_tokens=1)
        return math.ceil(per_slot / round_to) * round_to * concurrency
    need = required_tokens(per_request, concurrency, unit_tokens=1)
    return math.ceil(need / round_to) * round_to


# ---------------------------------------------------------------- vLLM /metrics

_SAMPLE_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)')
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> list[tuple[str, dict[str, str], float]]:
    """(name, labels, value) for every sample line of a Prometheus text exposition."""
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if m is None:
            continue
        labels = dict(_LABEL_RE.findall(m.group(2) or ""))
        try:
            samples.append((m.group(1), labels, float(m.group(3))))
        except ValueError:
            continue
    return samples


def _sum(samples, name: str, **match: str) -> Optional[float]:
    values = [v for n, labels, v in samples
              if n == name and all(labels.get(k) == want for k, want in match.items())]
    return sum(values) if values else None


def vllm_capacity(metrics_text: str) -> dict[str, Any]:
    """KV capacity from `vllm:cache_config_info`, whose labels are the whole
    CacheConfig (vLLM 0.31.0 vllm/v1/metrics/loggers.py)."""
    for name, labels, _ in parse_prometheus(metrics_text):
        if name == "vllm:cache_config_info" and labels.get("num_gpu_blocks") not in (None, "None"):
            blocks = int(labels["num_gpu_blocks"])
            block_size = int(labels["block_size"])
            return {"unit": "block", "unit_tokens": block_size, "units": blocks, "tokens": blocks * block_size}
    raise ValueError("vllm:cache_config_info with num_gpu_blocks not found in /metrics")


def vllm_counters(metrics_text: str) -> dict[str, Any]:
    """The cumulative values the post-check takes deltas of."""
    s = parse_prometheus(metrics_text)
    preemptions = _sum(s, "vllm:num_preemptions_total")
    if preemptions is None:
        preemptions = _sum(s, "vllm:num_preemptions")
    if preemptions is None:
        raise ValueError("vllm:num_preemptions not found in /metrics")
    return {"preemptions": preemptions, **_queue_time(s, "vllm:request_queue_time_seconds")}


def _queue_time(s, histogram: str) -> dict[str, Any]:
    buckets: dict[str, float] = {}
    for name, labels, value in s:
        if name == histogram + "_bucket":
            buckets[labels["le"]] = buckets.get(labels["le"], 0.0) + value
    return {"queue_time_sum_s": _sum(s, histogram + "_sum") or 0.0,
            "queue_time_count": _sum(s, histogram + "_count") or 0.0,
            "queue_time_buckets": buckets}


def vllm_counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "preemptions": int(after["preemptions"] - before["preemptions"]),
        "queue_time": {
            "sum_s": round(after["queue_time_sum_s"] - before["queue_time_sum_s"], 4),
            "count": int(after["queue_time_count"] - before["queue_time_count"]),
            "buckets": {le: int(v - before["queue_time_buckets"].get(le, 0.0))
                        for le, v in after["queue_time_buckets"].items()},
        },
    }


_VLLM_MAX_CONCURRENCY_RE = re.compile(r"[Mm]aximum concurrency for [\d,]+ tokens per request:\s*([\d.,]+)x")


def startup_log_max_concurrency(startup_log: str) -> Optional[float]:
    """The startup log's 'Maximum concurrency' figure, kept only to cross-check
    /metrics (ADR-0005 4). The log is for humans and its wording changes
    between versions, so a miss is None, never an error."""
    found = _VLLM_MAX_CONCURRENCY_RE.findall(startup_log)
    return float(found[-1].replace(",", "")) if found else None


# ---------------------------------------------------------------- SGLang /metrics

def sglang_capacity(metrics_text: str) -> dict[str, Any]:
    """KV capacity from the gauges SGLang sets once at startup
    (`sglang:max_total_num_tokens`, `sglang:page_size`; v0.5.21
    srt/observability/metrics_collector.py). Needs --enable-metrics."""
    s = parse_prometheus(metrics_text)
    tokens = _sum(s, "sglang:max_total_num_tokens")
    if tokens is None:
        raise ValueError("sglang:max_total_num_tokens not found in /metrics (start with --enable-metrics)")
    page = int(_sum(s, "sglang:page_size") or 1)
    return {"unit": "page", "unit_tokens": page, "units": int(tokens) // page, "tokens": int(tokens)}


def sglang_counters(metrics_text: str) -> dict[str, Any]:
    """The cumulative values the post-check takes deltas of: SGLang retracts
    (sends back to the queue) a running request when the KV cache runs out,
    its counterpart of vLLM's preemption."""
    s = parse_prometheus(metrics_text)
    retracted = _sum(s, "sglang:num_retracted_requests_total")
    if retracted is None:
        # A labelled counter has no sample until its first increment
        # (srt/observability/metrics_collector.py, increment_retracted_reqs, 0.5.21);
        # its sibling gauge is always there, so its presence means "none yet".
        if _sum(s, "sglang:num_retracted_reqs") is None:
            raise ValueError("neither sglang:num_retracted_requests_total nor "
                             "sglang:num_retracted_reqs found in /metrics")
        retracted = 0.0
    # Stored under vLLM's key, so one delta function serves both engines.
    return {"preemptions": retracted, **_queue_time(s, "sglang:queue_time_seconds")}


# A retraction is SGLang's preemption; the delta has the same shape as vLLM's.
sglang_counter_delta = vllm_counter_delta


# ---------------------------------------------------------------- TensorRT-LLM /metrics

def trtllm_capacity(iteration_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """KV capacity from trtllm-serve's /metrics, a JSON list of iteration
    statistics (needs enable_iter_perf_stats). Each carries kvCacheStats
    (maxNumBlocks, tokensPerBlock; v1.2.1 cpp/tensorrt_llm/executor/
    jsonSerialization.cpp). The list is empty until the executor has run an
    iteration, so the caller sends one short request first."""
    for stat in reversed(iteration_stats):
        kv = stat.get("kvCacheStats")
        if kv and kv.get("maxNumBlocks"):
            blocks, per_block = int(kv["maxNumBlocks"]), int(kv["tokensPerBlock"])
            return {"unit": "block", "unit_tokens": per_block, "units": blocks, "tokens": blocks * per_block}
    raise ValueError("no iteration statistics with kvCacheStats in /metrics "
                     "(enable_iter_perf_stats, and at least one request served)")


def trtllm_active(iteration_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """How many requests ran at once, from iteration statistics polled during a
    cell (numActiveRequests, numQueuedRequests; same file as kvCacheStats). Under
    GUARANTEED_NO_EVICT a request that does not fit waits in the queue silently,
    so this is the post-check's evidence that the concurrency was real."""
    active = [int(s.get("numActiveRequests") or 0) for s in iteration_stats]
    queued = [int(s.get("numQueuedRequests") or 0) for s in iteration_stats]
    return {"iterations": len(iteration_stats), "max_active": max(active, default=0),
            "max_queued": max(queued, default=0)}


# ---------------------------------------------------------------- llama.cpp /props

def llama_capacity(props: dict[str, Any], ctx_size: int, kv: str = "split") -> dict[str, Any]:
    """KV capacity of a server started with --ctx-size ctx_size.

    Split (--no-kv-unified): every slot holds its own cache of /props n_ctx
    tokens, so the total is n_ctx x total_slots, all from what the server reports.

    Unified (--kv-unified): llama-server reports no total: /props n_ctx is the
    per-slot limit. It equals
    the total only while the total is within the model's training context
    (Qwen3-8B: 40,960); past that the server caps each slot there and keeps the
    whole shared cache (2026-10-21, b11716 and b11785). So the total is the
    --ctx-size we set, the documented meaning of --kv-unified, and /props n_ctx
    is kept as the per-slot limit every request must also fit in."""
    if kv == "split":
        per_slot = int(props["default_generation_settings"]["n_ctx"])
        slots = int(props.get("total_slots", 0))
        return {"unit": "token", "unit_tokens": 1, "units": per_slot * slots, "tokens": per_slot * slots,
                "per_slot_tokens": per_slot, "slots": slots or None}
    return {"unit": "token", "unit_tokens": 1, "units": ctx_size, "tokens": ctx_size,
            "per_slot_tokens": int(props["default_generation_settings"]["n_ctx"]),
            "slots": int(props.get("total_slots", 0)) or None}


# ---------------------------------------------------------------- HTTP

def fetch_metrics(base_url: str) -> str:
    import requests
    response = requests.get(f"{base_url.rstrip('/')}/metrics", timeout=30)
    response.raise_for_status()
    return response.text


def fetch_json(base_url: str, path: str) -> Any:
    import requests
    response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_props(base_url: str) -> dict[str, Any]:
    import requests
    response = requests.get(f"{base_url.rstrip('/')}/props", timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------- the checks

def pre_check(engine: str, capacity: dict[str, Any], max_prompt_tokens: int, max_tokens: int,
              concurrency: int) -> dict[str, Any]:
    per_request = per_request_max(max_prompt_tokens, max_tokens)
    required = required_tokens(per_request, concurrency, capacity["unit_tokens"])
    fits = required <= capacity["tokens"]
    if capacity.get("per_slot_tokens") is not None:
        # llama.cpp: the longest request must also fit in one slot.
        fits = fits and required_tokens(per_request, 1, capacity["unit_tokens"]) <= capacity["per_slot_tokens"]
    return {
        "engine": engine,
        "capacity": capacity,
        "required": {"concurrency": concurrency, "per_request_tokens": per_request,
                     "margin_units": MARGIN_UNITS, "tokens": required},
        "max_prompt_tokens_planned": max_prompt_tokens,
        "fits": fits,
    }


# Engines whose post-check is a /metrics counter of requests sent back for
# lack of KV cache (vLLM: preemption, SGLang: retraction); the rest (llama.cpp,
# TensorRT-LLM, whose default scheduler GUARANTEED_NO_EVICT admits a request
# only when its whole cache is reserved) are checked from the raw output.
PREEMPTION_COUNTERS = {"vllm": (vllm_counters, vllm_counter_delta),
                       "sglang": (sglang_counters, sglang_counter_delta)}


def post_check(engine: str, planned_max_prompt: int, observed_prompt_tokens: Iterable[Optional[int]],
               vllm_delta: Optional[dict[str, Any]] = None, length_finished: int = 0,
               output_limit_finished: int = 0, trt_active: Optional[dict[str, Any]] = None,
               concurrency: Optional[int] = None) -> dict[str, Any]:
    """Whether the run stayed within what the pre-check assumed. Anything off
    means the estimate was wrong: the cell's numbers are not used for a
    verdict (ADR-0005 3). For llama.cpp, a response that stopped at the length
    limit after generating max_tokens hit its own output limit, which the
    pre-check reserved; only the rest (cut short by the context) count."""
    observed = [t for t in observed_prompt_tokens if t is not None]
    observed_max = max(observed) if observed else None
    result: dict[str, Any] = {"max_prompt_tokens_observed": observed_max}
    problems = []
    if observed_max is not None and observed_max > planned_max_prompt:
        problems.append(f"observed prompt {observed_max} > planned {planned_max_prompt}")
    if engine in PREEMPTION_COUNTERS:
        if vllm_delta is None:
            raise ValueError(f"the {engine} post-check needs the /metrics delta")
        result["preemptions"] = vllm_delta["preemptions"]
        result["queue_time"] = vllm_delta["queue_time"]
        if vllm_delta["preemptions"] > 0:
            problems.append(f"{vllm_delta['preemptions']} preemptions")
    else:
        result["length_finished"] = length_finished
        result["output_limit_finished"] = output_limit_finished
        context_limited = length_finished - output_limit_finished
        if context_limited > 0:
            problems.append(f"{context_limited} responses stopped at the length limit before max_tokens")
    if engine == "tensorrt_llm" and trt_active is not None:
        # docs/adr/0009 2: the concurrency reached the engine, not only its queue
        result["active_requests"] = trt_active
        if not trt_active["iterations"]:
            problems.append("no iteration statistics polled during the cell")
        elif concurrency is not None and trt_active["max_active"] < concurrency:
            problems.append(f"at most {trt_active['max_active']} requests ran at once (concurrency {concurrency})")
    result["ok"] = not problems
    result["problems"] = problems
    return result
