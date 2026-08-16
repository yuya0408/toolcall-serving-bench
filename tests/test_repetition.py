"""Smoke test for the repeated-trial harness: order-sensitivity produces
inconsistency, order-insensitivity does not, and the aggregate is sane."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval_bench.dataset.loader import load_cases
from eval_bench.dataset.tools import load_tools
from eval_bench.scorers.separated import Prediction
from eval_bench.repetition import aggregate_repeated, bootstrap_ci, run_repeated_case


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


if __name__ == "__main__":
    test_order_insensitive_predictor_is_perfectly_consistent()
    test_order_sensitive_predictor_is_detected_as_inconsistent()
    test_aggregate_and_bootstrap_ci_are_sane()
    test_bootstrap_ci_is_non_degenerate_for_mixed_accuracy()
