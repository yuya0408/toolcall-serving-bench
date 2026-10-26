"""scripts/noninferiority.py: the quantization non-inferiority verdict
(docs/adr/0008), on hand-made arms."""
from __future__ import annotations
import importlib.util
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("noninferiority", _ROOT / "scripts" / "noninferiority.py")
ni = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ni)

_RIGHT = {"tool": "aggregate_sales", "args": {"period": "last_month"}}
_WRONG = {"tool": "trend_inventory", "args": {}}
_TRIALS = 5


def _cases(n):
    return {f"c{i}": {"id": f"c{i}", "expected_tool": "aggregate_sales",
                      "expected_args": {"exact": {"period": "last_month"}, "semantic": {}},
                      "forbidden_tools": [], "negative": False} for i in range(n)}


def _arm(n, wrong):
    """Every (case, trial) right except the pairs in `wrong`."""
    return {(f"c{i}", t): {"decoded": _WRONG if (i, t) in wrong else _RIGHT, "output": None}
            for i in range(n) for t in range(_TRIALS)}


def test_identical_arms_are_outside_the_checked_range_so_not_shown():
    # no discordant pairs: p_d = 0 < 4.5%, where the calibration says nothing
    r = ni.noninferiority(_arm(200, set()), _arm(200, set()), _cases(200))
    assert r["selection_diff"] == 0 and r["p_d"] == 0
    assert not r["in_checked_range"] and r["verdict"] == "not shown"


def test_balanced_flips_spread_over_cases_are_non_inferior():
    # 50 discordant pairs (p_d 5%) on 50 cases, half each way: diff 0, rho ~ 0
    lost = {(i, 0) for i in range(25)}
    gained = {(i, 0) for i in range(25, 50)}
    r = ni.noninferiority(_arm(200, gained), _arm(200, lost), _cases(200))
    assert r["selection_diff"] == pytest.approx(0)
    assert r["p_d"] == pytest.approx(0.05) and r["rho"] <= ni.MAX_RHO
    assert r["lower_bound"] > -ni.MARGIN and r["verdict"] == "non-inferior"


def test_one_sided_losses_are_not_shown():
    # 50 pairs lost under 4bit and none gained: diff -5pt
    lost = {(i, 0) for i in range(50)}
    r = ni.noninferiority(_arm(200, set()), _arm(200, lost), _cases(200))
    assert r["selection_diff"] == pytest.approx(-50 / 1000)
    assert r["verdict"] == "not shown"


def test_flips_concentrated_on_few_cases_are_outside_the_checked_range():
    # the same 50 discordant pairs packed into 10 cases (all 5 trials): rho ~ 1
    lost = {(i, t) for i in range(5) for t in range(_TRIALS)}
    gained = {(i, t) for i in range(5, 10) for t in range(_TRIALS)}
    r = ni.noninferiority(_arm(200, gained), _arm(200, lost), _cases(200))
    assert r["rho"] > ni.MAX_RHO and not r["in_checked_range"]
    assert r["verdict"] == "not shown"


def test_arms_must_cover_the_same_pairs():
    with pytest.raises(ValueError):
        ni.noninferiority(_arm(3, set()), _arm(2, set()), _cases(3))


def test_rho_is_zero_when_discordance_is_spread_evenly():
    assert ni.intra_case_rho([1] * 50 + [0] * 150, _TRIALS) < 0
    assert ni.intra_case_rho([5] * 10 + [0] * 190, _TRIALS) == pytest.approx(1.0)
