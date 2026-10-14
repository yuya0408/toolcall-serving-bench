"""Smoke test for the repeated-trial harness: order-sensitivity produces
inconsistency, order-insensitivity does not, and the aggregate is sane."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from eval_bench.dataset.loader import load_cases
from eval_bench.dataset.tools import load_tools
from eval_bench.dataset.toolsets import build_tool_set
from eval_bench.scorers.separated import CaseScore, Prediction
from eval_bench.repetition import (
    RepeatedCaseResult,
    aggregate_repeated,
    bootstrap_ci,
    equivalence_verdict,
    half_width,
    run_repeated_case,
    run_repeated_cases_concurrent,
)
from eval_bench.repetition import _paired_cluster_bootstrap_diff_ci


def _judge(exp, pred):  # trivial semantic judge, mirrors tests/test_smoke.py
    return exp.strip() in pred or pred.strip() in exp


def _case_0101(cases):
    return next(c for c in cases if c["id"] == "case_0101")


def test_order_insensitive_predictor_is_perfectly_consistent():
    cases = load_cases("data")
    tools = load_tools()
    case = _case_0101(cases)

    def predict_always_correct(c, tool_set):
        exact = c["expected_args"].get("exact", {})
        semantic = c["expected_args"].get("semantic", {})
        return Prediction(tool=c["expected_tool"], args={**exact, **semantic})

    result = run_repeated_case(case, tools, stage_size=10, predict=predict_always_correct, judge=_judge, n=5)
    assert result.consistent is True
    assert all(s.selection for s in result.trials)


def test_order_sensitive_predictor_is_detected_as_inconsistent():
    cases = load_cases("data")
    tools = load_tools()
    case = _case_0101(cases)
    seen_sets = []

    def predict_first_in_list(c, tool_set):
        # Deliberately order-sensitive: "whatever the model saw first". This is
        # the failure mode consistency exists to catch.
        seen_sets.append(frozenset(tool_set))
        return Prediction(tool=tool_set[0], args={})

    result = run_repeated_case(case, tools, stage_size=10, predict=predict_first_in_list, judge=_judge, n=5)

    # The whole point of fixing the tool set once is that composition must not
    # drift between trials -- only order does.
    assert len(set(seen_sets)) == 1, "tool-set membership must be identical across trials"

    picks = {p.tool for p in result.predictions}
    assert len(picks) > 1, "shuffled order should change tool_set[0] across trials"
    assert result.consistent is False


def test_aggregate_and_bootstrap_ci_are_sane():
    cases = load_cases("data")
    tools = load_tools()

    def predict_always_correct(c, tool_set):
        if c["expected_tool"] is None:
            return Prediction(tool=None, args={})
        exact = c["expected_args"].get("exact", {})
        semantic = c["expected_args"].get("semantic", {})
        return Prediction(tool=c["expected_tool"], args={**exact, **semantic})

    results = [
        run_repeated_case(c, tools, stage_size=10, predict=predict_always_correct, judge=_judge, n=5)
        for c in cases
    ]
    agg = aggregate_repeated(results)
    assert agg["n_cases"] == len(cases)
    assert agg["n_trials_per_case"] == 5
    assert agg["selection_mean"] == 1.0
    assert agg["consistency"] == 1.0
    lo, hi = agg["selection_ci95"]
    assert 0.0 <= lo <= agg["selection_mean"] <= hi <= 1.0
    # predict_always_correct feeds back every expected exact/semantic field,
    # so both means should be perfect wherever they were applicable at all.
    if agg["arg_exact_mean"] is not None:
        assert agg["arg_exact_mean"] == 1.0
    if agg["arg_semantic_mean"] is not None:
        assert agg["arg_semantic_mean"] == 1.0

    lo2, hi2 = bootstrap_ci([1.0, 1.0, 1.0])
    assert lo2 == hi2 == 1.0


def test_bootstrap_ci_is_non_degenerate_for_mixed_accuracy():
    """A perfect predictor collapses the CI to a point regardless of whether
    the resampling is cluster-aware -- it doesn't prove the interval is
    computed correctly. This exercises the actual resampling with a
    predictor that is right on some cases and wrong on others."""
    cases = load_cases("data")
    tools = load_tools()
    correct_ids = {c["id"] for c in cases[: len(cases) // 2]}

    def predict_mixed(c, tool_set):
        if c["id"] in correct_ids:
            if c["expected_tool"] is None:
                return Prediction(tool=None, args={})
            exact = c["expected_args"].get("exact", {})
            semantic = c["expected_args"].get("semantic", {})
            return Prediction(tool=c["expected_tool"], args={**exact, **semantic})
        # Deliberately wrong: call some tool the case does not expect
        # (including calling a tool at all on a negative case).
        wrong = next(t for t in tool_set if t != c["expected_tool"])
        return Prediction(tool=wrong, args={})

    results = [
        run_repeated_case(c, tools, stage_size=10, predict=predict_mixed, judge=_judge, n=5)
        for c in cases
    ]
    agg = aggregate_repeated(results)
    assert 0.0 < agg["selection_mean"] < 1.0
    lo, hi = agg["selection_ci95"]
    assert hi - lo > 0, "CI must be non-degenerate when accuracy is not 0 or 1"
    print("repetition ok")


def test_concurrent_driver_matches_the_serialized_driver_trial_for_trial():
    """The accuracy_under_load probe's paired difference CI is only
    meaningful if a concurrent run and a serialized run over the same
    cases/n are trial-for-trial paired -- i.e. produce identical
    predictions when the predict function itself doesn't depend on
    concurrency. If run_repeated_cases_concurrent's shuffle seeds ever
    diverged from run_repeated_case's, this would catch it (the pairing
    tested in test_paired_diff_ci_is_narrower_than_unpaired_* assumes this
    holds; it doesn't re-verify it against the real driver)."""
    cases = load_cases("data")
    tools = load_tools()

    def predict_first_in_list(c, tool_set):
        return Prediction(tool=tool_set[0], args={})

    serialized = [
        run_repeated_case(c, tools, stage_size=10, predict=predict_first_in_list, judge=_judge, n=5)
        for c in cases
    ]
    concurrent = run_repeated_cases_concurrent(
        cases, tools, stage_size=10, predict=predict_first_in_list, judge=_judge, n=5, max_workers=4,
    )

    serialized_by_id = {r.case_id: r for r in serialized}
    concurrent_by_id = {r.case_id: r for r in concurrent}
    assert set(serialized_by_id) == set(concurrent_by_id)
    for case_id in serialized_by_id:
        assert [p.tool for p in serialized_by_id[case_id].predictions] == \
               [p.tool for p in concurrent_by_id[case_id].predictions]


def test_concurrent_driver_actually_overlaps_in_time():
    """Correctness alone (the test above) would also pass for an
    accidentally-serial implementation of run_repeated_cases_concurrent.
    This checks the thing that test can't: that requests are genuinely
    in flight at the same time, not dispatched one at a time under the
    hood."""
    import threading
    import time

    max_in_flight = 0
    current_in_flight = 0
    lock = threading.Lock()

    def predict_slow(c, tool_set):
        nonlocal max_in_flight, current_in_flight
        with lock:
            current_in_flight += 1
            max_in_flight = max(max_in_flight, current_in_flight)
        time.sleep(0.05)
        with lock:
            current_in_flight -= 1
        return Prediction(tool=None, args={})

    cases = load_cases("data")[:4]
    tools = load_tools()
    run_repeated_cases_concurrent(
        cases, tools, stage_size=10, predict=predict_slow, judge=_judge, n=2, max_workers=4,
    )
    assert max_in_flight > 1, "requests never overlapped -- driver is not actually concurrent"


def test_concurrent_driver_preserves_trial_order_even_when_completion_order_is_reversed():
    """Not a regression test for a live bug: ThreadPoolExecutor.map already
    guarantees its return order matches submission order regardless of
    completion order, so this passes against both the current
    trial-index-tracking implementation and the earlier version that
    grouped by return order without carrying the trial index explicitly
    (verified empirically by reverting the fix and re-running this test --
    it still passed, since map()'s ordering guarantee was already doing the
    work). Kept anyway as a property check that doesn't rely on the reader
    knowing that guarantee, and because carrying the trial index explicitly
    is more robust to a future change in how work_items gets built.
    Interleaving cases (trial-major instead of case-major submission order)
    would still be safe under the old return-order grouping, since map()'s
    guarantee keeps each case's own trials in ascending order wherever they
    fall in the stream -- but *shuffling* work_items (plausible, e.g. to
    avoid bursty per-case submission patterns hitting the server) would not
    be, since that can reorder one case's own trials relative to each
    other. Explicit trial tracking is immune to that; the old approach
    was not.
    """
    import random as stdlib_random
    import time

    case = load_cases("data")[0]
    tools = load_tools()
    stage_size = 10
    n = 5
    tool_set = build_tool_set(case, tools, stage_size, seed=0)

    # Ground truth, computed independently of the driver: which trial
    # produces which shuffled first tool.
    expected_first_tool_by_trial = {}
    for trial in range(n):
        order = list(tool_set)
        stdlib_random.Random(f"order:0:{case['id']}:{stage_size}:{trial}").shuffle(order)  # seed_base=0
        expected_first_tool_by_trial[trial] = order[0]
    first_tool_to_trial = {v: k for k, v in expected_first_tool_by_trial.items()}

    def predict_with_reversed_delay(c, ts):
        trial = first_tool_to_trial[ts[0]]
        time.sleep((n - trial) * 0.03)  # trial 0 sleeps longest -> finishes last
        return Prediction(tool=ts[0], args={})

    [result] = run_repeated_cases_concurrent(
        [case], tools, stage_size=stage_size, predict=predict_with_reversed_delay,
        judge=_judge, n=n, max_workers=n,
    )
    actual = [p.tool for p in result.predictions]
    expected = [expected_first_tool_by_trial[t] for t in range(n)]
    assert actual == expected, "predictions came out in completion order, not trial order"


def test_concurrent_driver_drops_no_work_items():
    cases = load_cases("data")
    tools = load_tools()

    def predict_always_correct(c, tool_set):
        if c["expected_tool"] is None:
            return Prediction(tool=None, args={})
        exact = c["expected_args"].get("exact", {})
        semantic = c["expected_args"].get("semantic", {})
        return Prediction(tool=c["expected_tool"], args={**exact, **semantic})

    results = run_repeated_cases_concurrent(
        cases, tools, stage_size=10, predict=predict_always_correct, judge=_judge, n=5, max_workers=8,
    )
    assert len(results) == len(cases)
    assert all(len(r.trials) == 5 for r in results)


def _fake_result(case_id, arg_exact_vals, arg_semantic_vals):
    trials = [
        CaseScore(case_id=case_id, selection=True, arg_exact=ae, arg_semantic=asem, e2e=False)
        for ae, asem in zip(arg_exact_vals, arg_semantic_vals)
    ]
    predictions = [Prediction(tool="t", args={}) for _ in trials]
    return RepeatedCaseResult(case_id=case_id, stage_size=10, trials=trials, predictions=predictions)


def test_arg_means_are_null_when_never_applicable():
    """A stage/model where no case has scoreable exact/semantic fields must
    report null, per run_manifest.schema.json's 'null iff no case had
    scoreable fields' contract -- not 0.0 or NaN, which would misread as
    'measured and failing'."""
    results = [_fake_result("c1", [None, None], [None, None])]
    agg = aggregate_repeated(results)
    assert agg["arg_exact_mean"] is None
    assert agg["arg_semantic_mean"] is None


def test_arg_means_pool_only_applicable_trials():
    # c1 has exact fields (3/4 correct across its trials), c2 has none (N/A).
    # c2's None entries must not dilute c1's rate.
    results = [
        _fake_result("c1", [True, True, True, False], [None, None, None, None]),
        _fake_result("c2", [None, None], [True, False]),
    ]
    agg = aggregate_repeated(results)
    assert agg["arg_exact_mean"] == 0.75
    assert agg["arg_semantic_mean"] == 0.5


def _fake_selection_result(case_id, selection_vals):
    trials = [
        CaseScore(case_id=case_id, selection=sel, arg_exact=None, arg_semantic=None, e2e=False)
        for sel in selection_vals
    ]
    predictions = [Prediction(tool="t" if sel else None, args={}) for sel in selection_vals]
    return RepeatedCaseResult(case_id=case_id, stage_size=10, trials=trials, predictions=predictions)


def test_paired_diff_ci_rejects_mismatched_case_count():
    baseline = [_fake_selection_result("c1", [True, True])]
    loaded = [_fake_selection_result("c1", [True, True]), _fake_selection_result("c2", [True, True])]
    try:
        _paired_cluster_bootstrap_diff_ci(baseline, loaded)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_paired_diff_ci_rejects_mismatched_case_order():
    baseline = [_fake_selection_result("c1", [True, True])]
    loaded = [_fake_selection_result("c2", [True, True])]
    try:
        _paired_cluster_bootstrap_diff_ci(baseline, loaded)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_paired_diff_ci_is_exactly_zero_when_conditions_are_identical():
    # Same underlying results in both arms: every paired resample must
    # produce a difference of exactly 0, so the CI collapses to a point.
    results = [_fake_selection_result(f"c{i}", [True, True, False, True, True]) for i in range(14)]
    lo, hi = _paired_cluster_bootstrap_diff_ci(results, results)
    assert lo == hi == 0.0


def test_paired_diff_ci_detects_a_real_degradation():
    baseline = [_fake_selection_result(f"c{i}", [True] * 5) for i in range(14)]
    loaded = [_fake_selection_result(f"c{i}", [False] * 5) for i in range(14)]
    lo, hi = _paired_cluster_bootstrap_diff_ci(baseline, loaded)
    assert lo == hi == -1.0  # loaded always wrong, baseline always right -> diff is always -1


def _unpaired_diff_ci(baseline, loaded, n_resamples=2000, alpha=0.05, seed=0):
    """Test-only stand-in for the mistake _paired_cluster_bootstrap_diff_ci
    exists to avoid: draws independent case-index resamples for each arm
    per iteration, instead of sharing one resampled index set across both.
    Used only to demonstrate that the paired version is narrower when cases
    are correlated -- see test_paired_diff_ci_is_narrower_than_unpaired_*.
    """
    import random
    rng = random.Random(seed)
    n = len(baseline)
    diffs = []
    for _ in range(n_resamples):
        base_idx = [rng.randrange(n) for _ in range(n)]
        load_idx = [rng.randrange(n) for _ in range(n)]
        base_vals = [1.0 if s.selection else 0.0 for i in base_idx for s in baseline[i].trials]
        load_vals = [1.0 if s.selection else 0.0 for i in load_idx for s in loaded[i].trials]
        diffs.append(sum(load_vals) / len(load_vals) - sum(base_vals) / len(base_vals))
    diffs.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)
    return diffs[lo_idx], diffs[hi_idx]


def test_paired_diff_ci_is_narrower_than_unpaired_when_cases_are_correlated():
    """The degenerate identical/all-flipped tests above pass whether or not
    resample indices are actually shared between arms -- they don't exercise
    pairing at all. This is the test that does: construct 14 cases with a
    large *between-case* spread (bimodal hard/easy) but a small, consistent
    *within-case* effect (one trial flipped per case). Pairing cancels the
    between-case composition noise out of the difference; not pairing lets
    it leak in. If this regressed to two independently-resampled index sets,
    this assertion -- not the degenerate ones above -- is what would fail.
    """
    baseline = []
    loaded = []
    for i in range(14):
        base_vals = [True, True, True, True, False] if i % 2 == 0 else [False, False, False, False, True]
        load_vals = list(base_vals)
        load_vals[0] = not load_vals[0]  # one trial flipped: same small effect on every case
        baseline.append(_fake_selection_result(f"c{i}", base_vals))
        loaded.append(_fake_selection_result(f"c{i}", load_vals))

    paired_lo, paired_hi = _paired_cluster_bootstrap_diff_ci(baseline, loaded, seed=0)
    unpaired_lo, unpaired_hi = _unpaired_diff_ci(baseline, loaded, seed=0)

    assert (paired_hi - paired_lo) < (unpaired_hi - unpaired_lo)


def test_half_width():
    assert half_width((0.3, 0.7)) == pytest.approx(0.2)


def test_equivalence_verdict_three_way():
    assert equivalence_verdict((-0.05, 0.05), margin=0.1) == "equivalent"
    assert equivalence_verdict((0.2, 0.3), margin=0.1) == "not_equivalent"
    assert equivalence_verdict((-0.3, -0.2), margin=0.1) == "not_equivalent"
    assert equivalence_verdict((-0.05, 0.15), margin=0.1) == "inconclusive"
    assert equivalence_verdict((-0.15, 0.05), margin=0.1) == "inconclusive"


def test_seed_base_reorders_but_keeps_the_presented_tool_set():
    """repetition.seed_base used to be recorded in the manifest without
    affecting anything (the shuffle seed ignored it). It now shifts every
    trial's order, while the presented tool set -- fixed by build_tool_set --
    stays the same, so runs differing only in seed_base are order-only
    variants of each other."""
    from eval_bench.repetition import trial_order

    case = load_cases("data")[0]
    tool_set = build_tool_set(case, load_tools(), 10, seed=0)

    same = trial_order(case["id"], tool_set, 10, trial=0, seed_base=0)
    assert same == trial_order(case["id"], tool_set, 10, trial=0, seed_base=0)

    orders = [trial_order(case["id"], tool_set, 10, trial=0, seed_base=b) for b in range(5)]
    assert len({tuple(o) for o in orders}) > 1
    assert all(sorted(o) == sorted(tool_set) for o in orders)


if __name__ == "__main__":
    test_order_insensitive_predictor_is_perfectly_consistent()
    test_order_sensitive_predictor_is_detected_as_inconsistent()
    test_aggregate_and_bootstrap_ci_are_sane()
    test_bootstrap_ci_is_non_degenerate_for_mixed_accuracy()
    test_arg_means_are_null_when_never_applicable()
    test_arg_means_pool_only_applicable_trials()
    test_paired_diff_ci_rejects_mismatched_case_count()
    test_paired_diff_ci_rejects_mismatched_case_order()
    test_paired_diff_ci_is_exactly_zero_when_conditions_are_identical()
    test_paired_diff_ci_detects_a_real_degradation()
    test_paired_diff_ci_is_narrower_than_unpaired_when_cases_are_correlated()
    test_concurrent_driver_matches_the_serialized_driver_trial_for_trial()
    test_concurrent_driver_actually_overlaps_in_time()
    test_concurrent_driver_preserves_trial_order_even_when_completion_order_is_reversed()
    test_concurrent_driver_drops_no_work_items()
    test_half_width()
    test_equivalence_verdict_three_way()
    test_seed_base_reorders_but_keeps_the_presented_tool_set()

