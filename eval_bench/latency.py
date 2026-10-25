"""The latency pass (docs/adr/0003-latency-pass-verdict.md).

One cell = one engine x quantization x tool_stage x concurrency. Within it:

  1. `concurrency` workers send requests back to back (closed loop). Request
     i is case i mod len(cases), with that case's stage tool set in its fixed
     seed-0 order and a per-request identifier appended to the query, so the
     tool definitions hit the prefix cache and the query never does.
  2. Warmup is one cycle of the case set (len(cases) completions): enough
     for every engine in the pilot to settle (ADR-0003, 2026-10-23).
  3. The next `n_requests` requests sent after that point are measured. The
     workers keep sending until every measured request has finished, so each
     one ran under the full concurrency (the extra ones are logged as
     cooldown and not counted). Whether the measured span was steady is
     checked afterwards with NVIDIA Perf Analyzer's rule in its count_windows
     form: a window is one cycle of the case set in the order sent (so every
     window carries the same query mix), and every measured window's throughput and mean latency
     must have max/min within 10%. An unsteady cell is flagged, not dropped:
     slowing down under sustained load is how the engine behaves.
  4. Each measured request either meets a latency class's TTFT and TPOT
     limits or exceeds them (DistServe's goodput counting); a timeout is an
     exceedance. The exceedance rate is judged against 1% with one-sided 95%
     Clopper-Pearson bounds: meets / fails / indeterminate.

Errors other than a timeout (HTTP errors, a dead server, thinking output
despite enable_thinking=False) abort the cell: they mean this configuration
does not run, which ADR-0003 records as "unsupported", not as a tail-latency
data point.
"""
from __future__ import annotations
import math
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ADR-0003 table: MLPerf Inference's Llama 3.1 8B limits.
SLO_CLASSES: dict[str, dict[str, float]] = {
    "conversational": {"ttft_ms": 2000.0, "tpot_ms": 100.0},
    "interactive": {"ttft_ms": 500.0, "tpot_ms": 30.0},
}
EXCEEDANCE_LIMIT = 0.01      # judged at p99 (MLPerf Server's 99% tail)
CONFIDENCE = 0.95            # one-sided, per bound
MIN_REQUESTS = 299           # smallest n where 0 exceedances can show <= 1% (0.99^n <= 0.05)
# The main run's n, fixed after the pilot (ADR-0003, 2026-10-23): five cycles of
# the 200-case set; meets with >= 80% chance at a true rate of 0.3% and fails
# with >= 80% at 2%.
MAIN_N_REQUESTS = 1000

# NVIDIA Perf Analyzer defaults (--stability-percentage, the "recent 3
# measurements"). STABLE_WINDOWS is the fewest windows the check needs.
STABILITY_PCT = 10.0
STABLE_WINDOWS = 3
WARMUP_CYCLES = 1            # cycles of the case set sent before measuring (ADR-0003, 2026-10-23)


class CellAborted(RuntimeError):
    """This configuration did not run; record it as unsupported, not as data."""


@dataclass
class TimedResult:
    """One request's timing, as measured by the client.

    ttft_s is the arrival of the first streamed chunk that carries a choice
    (vllm bench serve's definition); first_delta_s is the first chunk that
    carries generated text or a tool-call fragment, kept so the two can be
    compared on real engines. cached_tokens is None for an engine without a
    prefix cache (transformers_hf).
    """
    status: str                       # "ok" | "timeout"
    total_s: float
    ttft_s: Optional[float] = None
    first_delta_s: Optional[float] = None
    output_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, str]] = field(default_factory=list)   # [{"name", "arguments"}], streamed
    finish_reason: Optional[str] = None

    @property
    def tpot_s(self) -> Optional[float]:
        """(total - TTFT) / (output tokens - 1), excluding TTFT as MLPerf and
        LLMPerf do. Undefined for fewer than 2 output tokens."""
        if self.status != "ok" or self.ttft_s is None or not self.output_tokens or self.output_tokens < 2:
            return None
        return (self.total_s - self.ttft_s) / (self.output_tokens - 1)


# A sender takes (case, request index) and returns that request's timing.
SendFn = Callable[[dict[str, Any], int], TimedResult]


def new_run_tag() -> str:
    """Eight random digits, one per cell. Qwen3's tokenizer splits digits one
    per token, so every tag costs the same number of tokens. A random hex tag
    cost 16-22 tokens depending on how its letters merged, which made the
    prompts of two cells differ by a few tokens (2026-10-20)."""
    return f"{secrets.randbelow(10**8):08d}"


def tagged_query(query: str, run_tag: str, index: int) -> str:
    """The case query plus an identifier unique to this request (ADR-0003 4),
    so repeating a case never turns its whole prompt into a cache hit."""
    return f"{query}\n\n[request-id: {run_tag}-{index:06d}]"


# ---------------------------------------------------------------- statistics

def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    return min(1.0, sum(
        math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                 + i * log_p + (n - i) * log_q)
        for i in range(k + 1)
    ))


def _bisect(f: Callable[[float], float], lo: float = 0.0, hi: float = 1.0) -> float:
    """Root of a function that is decreasing in p, to well below 1e-9."""
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson_one_sided(k: int, n: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Exact one-sided (lower, upper) bounds on a binomial rate, each at
    `confidence`. No distributional assumption about latency itself."""
    if not 0 <= k <= n or n == 0:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    alpha = 1 - confidence
    upper = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p) - alpha)
    lower = 0.0 if k == 0 else _bisect(lambda p: alpha - (1 - _binom_cdf(k - 1, n, p)))
    return lower, upper


def verdict(k: int, n: int, limit: float = EXCEEDANCE_LIMIT) -> dict[str, Any]:
    lower, upper = clopper_pearson_one_sided(k, n)
    if upper <= limit:
        label = "meets"
    elif lower > limit:
        label = "fails"
    else:
        label = "indeterminate"
    return {"n": n, "exceeded": k, "rate_lower": lower, "rate_upper": upper, "verdict": label}


def percentiles(values: list[float]) -> dict[str, float]:
    """p50/p95/p99 by linear interpolation between order statistics (numpy's
    default), so the reported numbers can be re-derived from the raw log."""
    if not values:
        raise ValueError("no values")
    xs = sorted(values)

    def q(p: float) -> float:
        pos = (len(xs) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
    return {"p50": q(0.50), "p95": q(0.95), "p99": q(0.99)}


def exceeds(result: TimedResult, limits: dict[str, float]) -> bool:
    if result.status == "timeout":
        return True
    if result.ttft_s is None or result.ttft_s * 1000 > limits["ttft_ms"]:
        return True
    tpot = result.tpot_s
    return tpot is not None and tpot * 1000 > limits["tpot_ms"]


def is_stable(windows: list[dict[str, float]], pct: float = STABILITY_PCT,
              last: int = STABLE_WINDOWS) -> bool:
    """Perf Analyzer's rule: over the last `last` windows, max/min of both
    throughput and mean latency within `pct` percent."""
    if len(windows) < last:
        return False
    recent = windows[-last:]
    for key in ("throughput_rps", "mean_latency_s"):
        vals = [w[key] for w in recent]
        if min(vals) <= 0 or max(vals) / min(vals) > 1 + pct / 100:
            return False
    return True


# -------------------------------------------------------------------- driver

@dataclass
class _Row:
    index: int
    case_id: str
    sent_at: float
    done_at: float
    result: TimedResult


def run_latency_cell(cases: list[dict[str, Any]], send: SendFn, concurrency: int,
                     n_requests: int, clock: Callable[[], float] = time.perf_counter,
                     ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drive one cell; return (per-request log rows, the manifest's results).

    Raises CellAborted when the engine errors or did not start from an empty
    prefix cache. The rows collected so far are attached
    to the exception (`.rows`) so the caller can still keep the raw log.
    `clock` exists for tests; window throughput is measured on it.
    """
    window_size = len(cases)
    warmup_size = WARMUP_CYCLES * window_size
    lock = threading.Lock()
    stop = threading.Event()
    state: dict[str, Any] = {"next": 0, "measure_from": None, "error": None}
    rows: dict[int, _Row] = {}
    completions: list[_Row] = []
    t0 = clock()

    def on_complete(row: _Row) -> None:
        # Caller holds the lock.
        rows[row.index] = row
        completions.append(row)
        if state["measure_from"] is None and len(completions) == warmup_size:
            state["measure_from"] = state["next"]   # first request sent after this point
        start_idx = state["measure_from"]
        if start_idx is not None and all(i in rows for i in range(start_idx, start_idx + n_requests)):
            stop.set()

    def worker() -> None:
        while not stop.is_set():
            with lock:
                index = state["next"]
                state["next"] += 1
            case = cases[index % len(cases)]
            sent_at = clock()
            try:
                result = send(case, index)
            except Exception as exc:  # noqa: BLE001 -- any non-timeout failure aborts the cell
                with lock:
                    if state["error"] is None:
                        state["error"] = CellAborted(f"request {index} ({case['id']}): {exc!r}")
                stop.set()
                return
            with lock:
                on_complete(_Row(index, case["id"], sent_at - t0, clock(), result))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    log = [_log_row(rows[i], state["measure_from"], n_requests, t0) for i in sorted(rows)]
    if state["error"] is not None:
        state["error"].rows = log
        raise state["error"]

    try:
        _check_cache_started_cold(rows, concurrency)
    except CellAborted as exc:
        exc.rows = log
        raise
    start_idx = state["measure_from"]
    measured_rows = [rows[i] for i in range(start_idx, start_idx + n_requests)]
    measured = [r.result for r in measured_rows]
    after = start_idx + n_requests
    # The span ends when the request after it was sent; at concurrency 1 none
    # is sent, and the last measured completion is that same moment.
    end = rows[after].sent_at if after in rows else max(r.done_at for r in measured_rows) - t0
    windows = measured_windows([r.sent_at for r in measured_rows],
                               [r.result.total_s for r in measured_rows], end, window_size)
    results = summarize(measured, warmup={"window_requests": window_size, "warmup_requests": start_idx})
    results["measured_stability"] = {
        "window_requests": window_size,
        "windows": windows,
        "window_order": "sent",
        "stability_pct": STABILITY_PCT,
        # None: too few whole windows to judge (fewer than Perf Analyzer's 3).
        "stable": is_stable(windows, last=len(windows)) if len(windows) >= STABLE_WINDOWS else None,
    }
    return log, results


def measured_windows(sent_at: list[float], total_s: list[float], end: float,
                     window_size: int) -> list[dict[str, float]]:
    """The measured requests in the order they were sent, cut into whole
    windows of `window_size` (one case-set cycle, so every window holds the
    same cases); `end` is when the request after the measured span was sent.

    A window's throughput runs from its first send to the next window's first
    send (the last: to `end`). With a fixed number of requests in flight, a
    request is sent the moment one finishes, so sends per second equal
    completions per second. Completion order would put the slow requests of
    each cycle late: the first window would hold fewer of them and the last
    more (ADR-0003, 2026-10-25).
    """
    windows = []
    for start in range(0, len(sent_at) - window_size + 1, window_size):
        stop = start + window_size
        until = sent_at[stop] if stop < len(sent_at) else end
        windows.append({
            "throughput_rps": window_size / (until - sent_at[start]),
            "mean_latency_s": sum(total_s[start:stop]) / window_size,
        })
    return windows


def _log_row(row: _Row, measure_from: Optional[int], n_requests: int, t0: float) -> dict[str, Any]:
    if measure_from is None or row.index < measure_from:
        phase = "warmup"
    elif row.index < measure_from + n_requests:
        phase = "measured"
    else:
        phase = "cooldown"
    r = row.result
    ms = lambda s: None if s is None else round(s * 1000, 3)  # noqa: E731
    return {
        "index": row.index, "case_id": row.case_id, "phase": phase,
        "sent_at_s": round(row.sent_at, 4), "done_at_s": round(row.done_at - t0, 4),
        "status": r.status, "ttft_ms": ms(r.ttft_s), "first_delta_ms": ms(r.first_delta_s),
        "tpot_ms": ms(r.tpot_s), "total_ms": ms(r.total_s),
        "output_tokens": r.output_tokens, "prompt_tokens": r.prompt_tokens,
        "cached_tokens": r.cached_tokens, "content": r.content,
        "tool_calls": r.tool_calls, "finish_reason": r.finish_reason,
    }


def _check_cache_started_cold(rows: dict[int, _Row], concurrency: int) -> None:
    """The cell must start from an empty prefix cache (vLLM: reset at the cell
    boundary; llama.cpp: a freshly started server). Among the first
    `concurrency` requests -- sent together, so no single one is guaranteed
    to be scheduled first -- at least one must report 0 cached tokens."""
    first = [rows[i].result.cached_tokens for i in range(min(concurrency, len(rows))) if i in rows]
    if any(c is None for c in first):
        return  # engine without a prefix cache
    if min(first) != 0:
        raise CellAborted(
            f"the cell did not start from an empty prefix cache (first requests' cached tokens: "
            f"{first}); reset vLLM's cache or restart llama-server before the cell")


def summarize(measured: list[TimedResult], warmup: dict[str, Any]) -> dict[str, Any]:
    ok = [r for r in measured if r.status == "ok"]
    if not ok:
        raise CellAborted("every measured request timed out")
    tpots = [r.tpot_s * 1000 for r in ok if r.tpot_s is not None]
    results: dict[str, Any] = {
        "warmup_excluded": True,
        "n_requests": len(measured),
        "n_timeouts": len(measured) - len(ok),
        "ttft_ms": percentiles([r.ttft_s * 1000 for r in ok]),
        "tpot_ms": percentiles(tpots) if tpots else None,
        "total_latency_ms": percentiles([r.total_s * 1000 for r in ok]),
        "tokens_per_sec": percentiles([r.output_tokens / r.total_s for r in ok if r.output_tokens]),
        "slo": {name: {**limits, **verdict(sum(exceeds(r, limits) for r in measured), len(measured))}
                for name, limits in SLO_CLASSES.items()},
        "warmup": warmup,
    }
    cached = [r.cached_tokens for r in measured if r.cached_tokens is not None]
    if cached:
        prompt = sum(r.prompt_tokens or 0 for r in measured if r.cached_tokens is not None)
        results["prefix_cache"] = {
            "cached_tokens_mean": sum(cached) / len(cached),
            "hit_token_ratio": sum(cached) / prompt if prompt else 0.0,
        }
    return results
