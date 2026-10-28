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


def test_rho_is_low_when_discordance_is_spread_and_one_when_packed():
    assert ni.intra_case_rho([1] * 50 + [0] * 150, _TRIALS) < 0
    assert ni.intra_case_rho([5] * 10 + [0] * 190, _TRIALS) == pytest.approx(1.0)


@pytest.mark.parametrize("q,df,want", [
    (0.995, 199, 2.60076),   # scipy.stats.t.ppf, the case ADR-0008 uses
    (0.975, 199, 1.97196),
    (0.995, 30, 2.74999),
])
def test_t_quantile_matches_the_t_distribution(q, df, want):
    assert ni.t_quantile(q, df) == pytest.approx(want, abs=1e-3)


# docs/adr/0009 5 (option F), on scored arms: {(case, trial): 1 right / 0 wrong}

def _scored(n, wrong):
    return {(f"c{i}", t): int((i, t) not in wrong) for i in range(n) for t in range(_TRIALS)}


def test_pooled_identical_arms_are_non_inferior_without_a_range_rule():
    # ADR-0008 says 'not shown' here (p_d 0); option F has no range rule
    r = ni.pooled_noninferiority([_scored(200, set()), _scored(200, set())], _scored(200, set()))
    assert r["selection_diff"] == 0 and r["lower_bound"] == 0 and r["verdict"] == "non-inferior"
    assert r["p_d_vs_each_reference"] == [0, 0]


def test_pooled_uses_the_mean_of_the_references():
    # the references split on 40 pairs: their mean is 0.5 there; the test arm is right on all
    a = _scored(200, {(i, 0) for i in range(40)})
    b = _scored(200, set())
    r = ni.pooled_noninferiority([a, b], _scored(200, set()))
    assert r["selection_diff"] == pytest.approx(20 / 1000)
    assert r["p_d_vs_each_reference"] == [pytest.approx(0.04), 0]


def test_pooled_losses_beyond_the_margin_are_not_shown():
    lost = {(i, 0) for i in range(50)}   # -5pt
    r = ni.pooled_noninferiority([_scored(200, set())], _scored(200, lost))
    assert r["verdict"] == "not shown" and r["confidence"] == 0.999


def _arms(bad=()):
    """All nine configs right everywhere, except the names in `bad` lose 60 pairs (-6pt)."""
    lost = {(i, 0) for i in range(60)}
    return {f"{e}-{q}": _scored(200, lost if f"{e}-{q}" in bad else set())
            for e in ni.BATCHING_ENGINES for q in ("bf16",) + ni.BATCHING_QUANT}


def test_gate_bf16_references_are_the_other_bf16s_and_fixed():
    g = ni.batching_gate(_arms(bad={"trt-bf16"}))
    assert g["bf16"]["vllm-bf16"]["references"] == ["sglang-bf16", "trt-bf16"]   # trt kept though it fails
    assert g["bf16"]["trt-bf16"]["verdict"] == "not shown"
    assert g["passed_bf16"] == ["vllm-bf16", "sglang-bf16"]


def test_gate_quantized_is_judged_against_passed_bf16s_even_if_its_own_failed():
    g = ni.batching_gate(_arms(bad={"trt-bf16"}))
    r = g["quantized"]["trt-fp8"]
    assert r["references"] == ["vllm-bf16", "sglang-bf16"] and r["verdict"] == "non-inferior"
    assert r["diff_vs_own_bf16"] == pytest.approx(60 / 1000)   # descriptive only
    assert "trt-fp8" in g["passing"] and "trt-bf16" not in g["passing"]


def test_gate_lone_bf16_is_the_reference():
    arms = {k: v for k, v in _arms().items() if k.startswith("vllm-")}
    g = ni.batching_gate(arms)
    assert g["bf16"]["vllm-bf16"]["verdict"] == "reference"
    assert g["quantized"]["vllm-awq"]["references"] == ["vllm-bf16"]


def test_gate_nothing_passes_when_no_bf16_does():
    # two engines whose bf16s disagree on half the cases, in opposite directions:
    # each is 50pt below the other there, so neither is non-inferior to the other
    half = {(i, t) for i in range(100) for t in range(_TRIALS)}
    other = {(i, t) for i in range(100, 200) for t in range(_TRIALS)}
    arms = {"vllm-bf16": _scored(200, half), "sglang-bf16": _scored(200, other),
            "vllm-awq": _scored(200, set())}
    g = ni.batching_gate(arms)
    assert g["passed_bf16"] == [] and g["passing"] == []
    assert g["quantized"]["vllm-awq"]["verdict"] == "not shown (no bf16 passed)"


def _status(tmp_path, rows):
    (tmp_path / "status.jsonl").write_text("".join(f'{{"name": "{n}", "outcome": "{o}"}}\n' for n, o in rows))


def test_batching_loader_refuses_a_failed_arm_unless_excluded(tmp_path):
    # docs/adr/0009 5: a failed ni cell is rerun before the gate; only a repeat failure is left out, by name.
    _status(tmp_path, [("ni-vllm-bf16-c1", "error")])
    with pytest.raises(ValueError, match="rerun it"):
        ni.load_batching_arms([tmp_path], {})
    assert ni.load_batching_arms([tmp_path], {}, exclude=("vllm-bf16",)) == {}


def test_batching_loader_uses_the_last_run_and_needs_its_raw_log(tmp_path):
    _status(tmp_path, [("ni-trt-awq-c1", "error"), ("ni-trt-awq-c1", "ok")])
    (tmp_path / "ni-trt-awq-c1").mkdir()
    (tmp_path / "ni-trt-awq-c1" / "x.raw.partial.jsonl").write_text("")   # a failed run's partial log is not read
    with pytest.raises(ValueError, match="0 raw logs"):
        ni.load_batching_arms([tmp_path], {})


def test_batching_loader_needs_the_status(tmp_path):
    with pytest.raises(ValueError, match="status.jsonl"):
        ni.load_batching_arms([tmp_path], {})


def test_batching_loader_takes_a_rerun_from_a_later_folder(tmp_path):
    main, rerun = tmp_path / "main", tmp_path / "rerun"
    main.mkdir(); rerun.mkdir()
    _status(main, [("ni-vllm-bf16-c1", "error")])
    _status(rerun, [("ni-vllm-bf16-c1", "ok")])
    with pytest.raises(ValueError, match=r"rerun.ni-vllm-bf16-c1: 0 raw logs"):
        ni.load_batching_arms([main, rerun], {})
