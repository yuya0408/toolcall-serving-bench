"""The verdict path computes its statistics with the standard library only;
these check them against scipy/numpy over the ranges the verdicts use."""
from __future__ import annotations
import importlib.util
import pathlib
import random

import pytest

np = pytest.importorskip("numpy")
stats = pytest.importorskip("scipy.stats")

from eval_bench.latency import clopper_pearson_one_sided, percentiles  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("noninferiority", _ROOT / "scripts" / "noninferiority.py")
ni = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ni)


@pytest.mark.parametrize("n", [299, 1000])
@pytest.mark.parametrize("k", [0, 1, 5, 10, 24, 59, 299])
def test_clopper_pearson_matches_the_beta_quantiles(n, k):
    lower, upper = clopper_pearson_one_sided(k, n, confidence=0.95)
    want_lower = 0.0 if k == 0 else stats.beta.ppf(0.05, k, n - k + 1)
    want_upper = 1.0 if k == n else stats.beta.ppf(0.95, k + 1, n - k)
    assert lower == pytest.approx(want_lower, abs=1e-12)
    assert upper == pytest.approx(want_upper, abs=1e-12)


@pytest.mark.parametrize("q", [0.975, 0.995, 0.999])
@pytest.mark.parametrize("df", [30, 199, 999])
def test_t_quantile_matches_scipy(q, df):
    assert ni.t_quantile(q, df) == pytest.approx(stats.t.ppf(q, df), abs=1e-4)


def test_percentiles_match_numpy():
    rng = random.Random(0)
    xs = [rng.random() for _ in range(1001)]
    got = percentiles(xs)
    for name, p in (("p50", 50), ("p95", 95), ("p99", 99)):
        assert got[name] == pytest.approx(float(np.percentile(xs, p)), abs=1e-12)
