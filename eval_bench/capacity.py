"""Whether a concurrency>1 cell actually ran concurrently (docs/adr/0005).

A cell at concurrency c is only a measurement of "c requests at once" if the
engine's KV cache holds c requests even when every one of them grows to the
longest it can be (the guarantee-style definition, ADR-0005 2). This module
does the arithmetic and reads the engines' machine-facing endpoints:

- the pre-check, before the cell runs: c x (the longest request, rounded up
  to the engine's allocation unit) <= the KV capacity the server reports
  (vLLM: /metrics `vllm:cache_config_info`; llama.cpp: /props `n_ctx`);
- the post-check, after: vLLM preempted nothing (the `vllm:num_preemptions`
  counter's delta); llama.cpp, which has no such counter, returned no
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

# One allocation unit on top of the counted length, to absorb tokenizer
# differences between engines (ADR-0005 2). The offline count matched vLLM's
# prompt_tokens exactly and llama.cpp's was 4 tokens lower (2026-10-18 smoke).
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


def per_request_max(max_prompt_tokens: int, max_tokens: int) -> int:
    return max_prompt_tokens + max_tokens


def required_tokens(per_request: int, concurrency: int, unit_tokens: int,
                    margin_units: int = MARGIN_UNITS) -> int:
    """c x the longest request, each rounded up to whole allocation units."""
    units = math.ceil((per_request + margin_units * unit_tokens) / unit_tokens)
    return units * unit_tokens * concurrency


def llama_ctx_size(per_request: int, concurrency: int, round_to: int = 256) -> int:
    """llama.cpp's --ctx-size for a cell: exactly the requirement (ADR-0005 1),
    rounded up so the server's own padding cannot leave it short."""
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
    buckets: dict[str, float] = {}
    for name, labels, value in s:
        if name == "vllm:request_queue_time_seconds_bucket":
            buckets[labels["le"]] = buckets.get(labels["le"], 0.0) + value
    return {
        "preemptions": preemptions,
        "queue_time_sum_s": _sum(s, "vllm:request_queue_time_seconds_sum") or 0.0,
        "queue_time_count": _sum(s, "vllm:request_queue_time_seconds_count") or 0.0,
        "queue_time_buckets": buckets,
    }


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


# ---------------------------------------------------------------- llama.cpp /props

def llama_capacity(props: dict[str, Any]) -> dict[str, Any]:
    """KV capacity from /props. With --kv-unified every slot can use the whole
    shared cache, so the per-slot n_ctx is the total."""
    n_ctx = int(props["default_generation_settings"]["n_ctx"])
    return {"unit": "token", "unit_tokens": 1, "units": n_ctx, "tokens": n_ctx,
            "slots": int(props.get("total_slots", 0)) or None}


# ---------------------------------------------------------------- HTTP

def fetch_metrics(base_url: str) -> str:
    import requests
    response = requests.get(f"{base_url.rstrip('/')}/metrics", timeout=30)
    response.raise_for_status()
    return response.text


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
    return {
        "engine": engine,
        "capacity": capacity,
        "required": {"concurrency": concurrency, "per_request_tokens": per_request,
                     "margin_units": MARGIN_UNITS, "tokens": required},
        "max_prompt_tokens_planned": max_prompt_tokens,
        "fits": required <= capacity["tokens"],
    }


def post_check(engine: str, planned_max_prompt: int, observed_prompt_tokens: Iterable[Optional[int]],
               vllm_delta: Optional[dict[str, Any]] = None, length_finished: int = 0) -> dict[str, Any]:
    """Whether the run stayed within what the pre-check assumed. Anything off
    means the estimate was wrong: the cell's numbers are not used for a
    verdict (ADR-0005 3)."""
    observed = [t for t in observed_prompt_tokens if t is not None]
    observed_max = max(observed) if observed else None
    result: dict[str, Any] = {"max_prompt_tokens_observed": observed_max}
    problems = []
    if observed_max is not None and observed_max > planned_max_prompt:
        problems.append(f"observed prompt {observed_max} > planned {planned_max_prompt}")
    if engine == "vllm":
        if vllm_delta is None:
            raise ValueError("the vLLM post-check needs the /metrics delta")
        result["preemptions"] = vllm_delta["preemptions"]
        result["queue_time"] = vllm_delta["queue_time"]
        if vllm_delta["preemptions"] > 0:
            problems.append(f"{vllm_delta['preemptions']} preemptions")
    else:
        result["length_finished"] = length_finished
        if length_finished > 0:
            problems.append(f"{length_finished} responses stopped at the length limit")
    result["ok"] = not problems
    result["problems"] = problems
    return result
