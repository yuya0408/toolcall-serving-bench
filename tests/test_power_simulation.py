"""scripts/power_simulation.py must analyze simulated data the way the third
pass analyzes real data (repetition._paired_cluster_bootstrap_diff_ci), or
the case count it recommends answers a different question."""
from __future__ import annotations
import importlib.util
import pathlib

import pytest

np = pytest.importorskip("numpy")

from eval_bench.repetition import RepeatedCaseResult, _paired_cluster_bootstrap_diff_ci  # noqa: E402
from eval_bench.scorers.separated import CaseScore, Prediction  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("power_simulation", _ROOT / "scripts" / "power_simulation.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)


def _arm(case_id: str, selections: list[bool]) -> RepeatedCaseResult:
    trials = [CaseScore(case_id=case_id, selection=s, arg_exact=None, arg_semantic=None, e2e=False)
              for s in selections]
    return RepeatedCaseResult(case_id=case_id, stage_size=10, trials=trials,
                              predictions=[Prediction(tool=None, args={}) for _ in selections])


def test_simulated_ci_matches_the_real_paired_bootstrap():
    rng = np.random.default_rng(7)
    diffs = sim.simulate_case_diffs(rng, n_cases=60, p_d=0.15, rho=0.3, delta=0.0)

    # Turn each case's difference into two arms of 5 trials with that difference.
    baseline, loaded = [], []
    for i, d in enumerate(diffs):
        k = int(round(d * sim.TRIALS))
        base = [True] * sim.TRIALS
        load = [True] * sim.TRIALS
        for t in range(abs(k)):
            (base if k > 0 else load)[t] = False
        baseline.append(_arm(f"c{i}", base))
        loaded.append(_arm(f"c{i}", load))

    real = _paired_cluster_bootstrap_diff_ci(baseline, loaded, n_resamples=20000)
    fast = sim.bootstrap_ci(np.random.default_rng(0), diffs, n_resamples=20000)
    assert fast == pytest.approx(real, abs=0.005)   # same estimator; only the RNG streams differ


def test_true_difference_is_delta():
    rng = np.random.default_rng(0)
    diffs = sim.simulate_case_diffs(rng, n_cases=200000, p_d=0.2, rho=0.0, delta=-0.026)
    assert diffs.mean() == pytest.approx(-0.026, abs=0.002)


def test_impossible_delta_is_rejected():
    with pytest.raises(ValueError):
        sim.simulate_case_diffs(np.random.default_rng(0), 10, p_d=0.01, rho=0.0, delta=0.026)
