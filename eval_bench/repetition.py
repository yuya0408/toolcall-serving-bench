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
from typing import Any, Callable

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
        order = list(tool_set)
        random.Random(f"order:{case['id']}:{stage_size}:{trial}").shuffle(order)
        pred = predict(case, order)
        predictions.append(pred)
        trials.append(score_case(case, pred, judge))
    return RepeatedCaseResult(case_id=case["id"], stage_size=stage_size, trials=trials, predictions=predictions)


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


def aggregate_repeated(results: list[RepeatedCaseResult]) -> dict[str, Any]:
    """Aggregate across cases at one stage size: mean + cluster-bootstrap CI
    for selection, plus the case-level consistency rate.

    Metrics are pooled across all (case x trial) pairs -- this is the
    tool-count/degradation number. Consistency is computed per case (it is
    only meaningful within a case, across its own trials) and then averaged.
    """
    if not results:
        return {"stage_size": None, "n_cases": 0, "n_trials_per_case": 0,
                "selection_mean": float("nan"), "selection_ci95": (float("nan"), float("nan")),
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
        "consistency": sum(consistency_vals) / len(consistency_vals),
    }
