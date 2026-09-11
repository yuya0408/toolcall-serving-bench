"""Repeated-trial harness: separate the "tool count effect" from the
"presentation-order effect", and get consistency as a byproduct.

A single trial at a given stage size proves nothing about tool-count
degradation on its own -- a point estimate at n=1 has the shuffle order's
effect baked into it, and there is no way to tell whether a low score is
"too many tools" or "this particular ordering was unlucky" (position bias).

So each case is run `n` times per stage size, each trial with an
independently seeded tool-set order (`toolsets.build_tool_set` already makes
order a pure function of seed). Averaging the `n` trials gives a tool-count
estimate that isn't contaminated by order. Whether the *prediction* itself is
identical across all `n` trials is consistency: a model that flips its answer
when nothing but tool order changed is not safe to put in production, even
if its average accuracy looks fine. This falls out of the same trials the
CI needs -- no extra model calls.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

from eval_bench.dataset.toolsets import build_tool_set
from eval_bench.scorers.separated import CaseScore, Prediction, SemanticJudge, score_case

# Given a case and the tool names presented for one trial, produce a Prediction.
# This is where a real ModelAdapter + Decoder pipeline plugs in; kept as a plain
# callable so this module knows nothing about backends.
PredictFn = Callable[[dict[str, Any], list[str]], Prediction]


@dataclass
class RepeatedCaseResult:
    case_id: str
    stage_size: int
    trials: list[CaseScore]
    predictions: list[Prediction]

    @property
    def consistent(self) -> bool:
        """True iff every trial produced the identical (tool, args) prediction.

        Order changed between trials; the query, the tool-set membership, and
        the correct answer did not. Any divergence here is the model reacting
        to order, not to the task.

        Two definition choices are baked in, both strict by design (a model
        that is inconsistent by this measure is inconsistent by any looser
        one too), and both should be treated as fixed once ADR-0001's
        consistency threshold is calibrated against them:
          - unanimity across all n trials, not pairwise agreement. A case
            that is 4/5 identical counts as inconsistent, same as 1/5.
          - args compared by literal equality, including semantic (free-text)
            fields. Two trials judged semantically equivalent by the scorer's
            embedding judge can still count as inconsistent here if the
            surface text differs -- this metric is stricter than arg_semantic.
        """
        first = self.predictions[0]
        return all(p.tool == first.tool and p.args == first.args for p in self.predictions[1:])


def _run_one_trial(
    case: dict[str, Any],
    tool_set: list[str],
    stage_size: int,
    trial: int,
    predict: PredictFn,
    judge: SemanticJudge,
) -> tuple[Prediction, CaseScore]:
    """One (case, trial) unit of work: deterministically shuffle `tool_set`
    for this trial, call predict, score. Factored out of run_repeated_case
    so the serialized path and the concurrent load driver
    (run_repeated_cases_concurrent) derive their trial order from the exact
    same seed formula -- if they diverged, a baseline run and a loaded run
    over "the same" cases would no longer be trial-for-trial paired, and
    _paired_cluster_bootstrap_diff_ci's pairing would be nominal only.
    """
    order = list(tool_set)
    random.Random(f"order:{case['id']}:{stage_size}:{trial}").shuffle(order)
    pred = predict(case, order)
    return pred, score_case(case, pred, judge)


def run_repeated_case(
    case: dict[str, Any],
    tools: dict[str, dict[str, Any]],
    stage_size: int,
    predict: PredictFn,
    judge: SemanticJudge,
    n: int = 5,
) -> RepeatedCaseResult:
    """Run `n` trials of one case at one stage size, varying *only* the
    presentation order across trials.

    The tool set is built once (build_tool_set's seed also affects which
    padding tools are chosen, not just their order, so calling it once per
    trial with a different seed would confound "order changed" with "the
    presented tools changed" -- exactly the confound this harness exists to
    remove). Each trial then gets its own independently shuffled ordering of
    that same fixed set.
    """
    if n < 2:
        raise ValueError("n must be >= 2: consistency is undefined for a single trial")

    tool_set = build_tool_set(case, tools, stage_size, seed=0)

    trials: list[CaseScore] = []
    predictions: list[Prediction] = []
    for trial in range(n):
        pred, score = _run_one_trial(case, tool_set, stage_size, trial, predict, judge)
        predictions.append(pred)
        trials.append(score)
    return RepeatedCaseResult(case_id=case["id"], stage_size=stage_size, trials=trials, predictions=predictions)


def run_repeated_cases_concurrent(
    cases: list[dict[str, Any]],
    tools: dict[str, dict[str, Any]],
    stage_size: int,
    predict: PredictFn,
    judge: SemanticJudge,
    n: int,
    max_workers: int,
) -> list[RepeatedCaseResult]:
    """Same total workload as calling run_repeated_case once per case (what
    the serialized baseline does), but dispatches every (case, trial) unit
    across *all* cases through a thread pool with up to `max_workers` in
    flight at once. This is what "concurrency=N" means for the
    accuracy_under_load probe (docs/engine-performance-plan.md): N requests
    simultaneously in flight against the engine -- not N trials of one case.

    ThreadPoolExecutor, not asyncio: a ModelAdapter's generate() is a
    synchronous HTTP call (or an in-process transformers call), and HTTP I/O
    releases the GIL while waiting on the response, so a thread pool already
    achieves genuine concurrent requests against the server without an
    async rewrite of the adapter layer.

    Every trial's shuffle uses the exact same seed formula as
    run_repeated_case (via _run_one_trial), so this arm and a serialized
    run over the same cases/n/stage_size are trial-for-trial paired.
    """
    from concurrent.futures import ThreadPoolExecutor

    tool_sets = {c["id"]: build_tool_set(c, tools, stage_size, seed=0) for c in cases}
    work_items = [(c, trial) for c in cases for trial in range(n)]

    def _dispatch(item: tuple[dict[str, Any], int]) -> tuple[str, Prediction, CaseScore]:
        case, trial = item
        pred, score = _run_one_trial(case, tool_sets[case["id"]], stage_size, trial, predict, judge)
        return case["id"], pred, score

    # Executor.map returns results in submission order regardless of
    # completion order, and work_items groups each case's trials together
    # (in trial order) within that submission order -- so grouping by
    # case_id below reconstructs each case's trials in the right sequence.
    by_case: dict[str, list[tuple[Prediction, CaseScore]]] = {c["id"]: [] for c in cases}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for case_id, pred, score in pool.map(_dispatch, work_items):
            by_case[case_id].append((pred, score))

    results = []
    for c in cases:
        pairs = by_case[c["id"]]
        results.append(RepeatedCaseResult(
            case_id=c["id"], stage_size=stage_size,
            predictions=[p for p, _ in pairs], trials=[s for _, s in pairs],
        ))
    return results


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`, treating each value
    as an independent sample.

    General-purpose utility (e.g. for judge-threshold calibration on
    independent hand-labeled examples). `aggregate_repeated` does NOT use
    this for the selection metric -- trials within a case are correlated, so
    it uses `_cluster_bootstrap_selection_ci` instead. Use this one only when
    the values really are independent draws.
    """
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)
    return (means[lo_idx], means[hi_idx])


def _cluster_bootstrap_selection_ci(
    results: list[RepeatedCaseResult],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI for the pooled selection rate, resampling *cases* rather
    than individual trials.

    The n trials within a case are the same query under different orderings,
    not independent draws -- resampling all (case x trial) pairs as if they
    were independent understates the interval (effective n looks like
    n_cases * n_trials when it is really closer to n_cases). Since this CI is
    what a "degradation is real, not noise" claim rests on, an artificially
    tight interval would defeat the reason it exists.
    """
    if not results:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(results)
    means = []
    for _ in range(n_resamples):
        sample = [results[rng.randrange(n)] for _ in range(n)]
        vals = [1.0 if s.selection else 0.0 for r in sample for s in r.trials]
        means.append(sum(vals) / len(vals))
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)
    return (means[lo_idx], means[hi_idx])


def _paired_cluster_bootstrap_diff_ci(
    baseline: list[RepeatedCaseResult],
    loaded: list[RepeatedCaseResult],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI for the *paired difference* in pooled selection rate
    between two conditions measured over the same case set (e.g. the
    accuracy_under_load probe's concurrency=1 baseline vs. concurrency=16;
    see docs/engine-performance-plan.md's equivalence-margin section).

    Resamples case INDICES once per iteration and reuses that same index set
    for both `baseline` and `loaded` -- referencing both conditions from a
    single bootstrap draw is what makes this "paired" rather than two
    independent CIs whose overlap gets eyeballed, the mistake that section
    calls out by name (the same one vlm-vs-yolo's default-threshold critique
    was about). `baseline[i]` and `loaded[i]` must be the same case in the
    same order; this is checked, not re-derived from case_id.
    """
    if len(baseline) != len(loaded):
        raise ValueError(
            f"baseline has {len(baseline)} cases, loaded has {len(loaded)}: "
            "paired resampling requires the same case set, same order"
        )
    if not baseline:
        return (float("nan"), float("nan"))
    mismatched = [(b.case_id, l.case_id) for b, l in zip(baseline, loaded) if b.case_id != l.case_id]
    if mismatched:
        raise ValueError(f"baseline/loaded case order mismatch, e.g. {mismatched[0]}")

    rng = random.Random(seed)
    n = len(baseline)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        base_vals = [1.0 if s.selection else 0.0 for i in idx for s in baseline[i].trials]
        load_vals = [1.0 if s.selection else 0.0 for i in idx for s in loaded[i].trials]
        diffs.append(sum(load_vals) / len(load_vals) - sum(base_vals) / len(base_vals))
    diffs.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)
    return (diffs[lo_idx], diffs[hi_idx])


def half_width(ci: tuple[float, float]) -> float:
    """(hi - lo) / 2 -- the equivalence margin X is defined as this applied
    to the serialized baseline's own selection_ci95, per
    docs/engine-performance-plan.md (2026-09-11: corrected from full width,
    which had silently doubled the margin relative to the doc's own '±X pt'
    framing).
    """
    lo, hi = ci
    return (hi - lo) / 2


def equivalence_verdict(diff_ci: tuple[float, float], margin: float) -> str:
    """Classify a paired difference CI (from _paired_cluster_bootstrap_diff_ci)
    against equivalence margin X, TOST-style: the entire CI must sit inside
    (-margin, +margin) to call the two conditions equivalent -- not just the
    point estimate (docs/engine-performance-plan.md's option (a), chosen
    2026-09-11 specifically to avoid the weaker, direction-blind version
    where a wide, uncertain difference could pass just because its center
    happened to be small).

    Three outcomes, not two, because with a small case count (e.g. this
    project's n=14) "not equivalent" and "not detected" are genuinely
    different claims:
      - "equivalent":     the whole CI fits inside (-margin, +margin).
      - "not_equivalent": the whole CI is outside (-margin, +margin), in
                           either direction -- confidently more different
                           than the shuffle-order noise floor already is.
      - "inconclusive":   the CI straddles a margin boundary. Cannot rule
                           out a difference bigger than margin, but cannot
                           confirm one either. This is an expected, legitimate,
                           reportable outcome at n=14 (confirmed with the user
                           2026-09-11) -- it is not license to loosen margin
                           after seeing the data; the margin was pre-declared
                           as a rule, not a number, for exactly this reason.

    Boundary convention (a CI bound landing exactly on +-margin, which 2000
    resamples make unlikely in practice but should still be unambiguous):
    both comparisons are closed at margin, i.e. touching it without crossing
    is "still within" -- so a CI equal to exactly [-margin, +margin] is
    "equivalent", and a CI that merely touches margin from outside while
    still containing points inside it is "inconclusive", not "not_equivalent".
    """
    lo, hi = diff_ci
    if lo >= -margin and hi <= margin:
        return "equivalent"
    if lo > margin or hi < -margin:
        return "not_equivalent"
    return "inconclusive"


def _applicable_mean(values: list[Optional[bool]]) -> Optional[float]:
    """Mean over the trials where the metric was applicable (non-None),
    or None if it was never applicable -- matches the run manifest schema's
    'null iff no case had scoreable fields' contract for arg_exact/arg_semantic.
    """
    applicable = [1.0 if v else 0.0 for v in values if v is not None]
    if not applicable:
        return None
    return sum(applicable) / len(applicable)


def aggregate_repeated(results: list[RepeatedCaseResult]) -> dict[str, Any]:
    """Aggregate across cases at one stage size: mean + cluster-bootstrap CI
    for selection, arg_exact/arg_semantic means, plus the case-level
    consistency rate.

    Metrics are pooled across all (case x trial) pairs -- this is the
    tool-count/degradation number. arg_exact/arg_semantic are averaged only
    over trials where they were applicable (selection succeeded on a
    positive case and the case has scoreable fields of that kind); see
    CaseScore/score_case in scorers/separated.py. Consistency is computed
    per case (it is only meaningful within a case, across its own trials)
    and then averaged.
    """
    if not results:
        return {"stage_size": None, "n_cases": 0, "n_trials_per_case": 0,
                "selection_mean": float("nan"), "selection_ci95": (float("nan"), float("nan")),
                "arg_exact_mean": None, "arg_semantic_mean": None,
                "consistency": float("nan")}

    all_trials = [s for r in results for s in r.trials]
    selection_vals = [1.0 if s.selection else 0.0 for s in all_trials]
    consistency_vals = [1.0 if r.consistent else 0.0 for r in results]

    return {
        "stage_size": results[0].stage_size,
        "n_cases": len(results),
        "n_trials_per_case": len(results[0].trials),
        "selection_mean": sum(selection_vals) / len(selection_vals),
        "selection_ci95": _cluster_bootstrap_selection_ci(results),
        "arg_exact_mean": _applicable_mean([s.arg_exact for s in all_trials]),
        "arg_semantic_mean": _applicable_mean([s.arg_semantic for s in all_trials]),
        "consistency": sum(consistency_vals) / len(consistency_vals),
    }
