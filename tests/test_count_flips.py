"""scripts/count_flips.py: the third pass's flip count (docs/adr/0002), on
hand-made raw logs."""
from __future__ import annotations
import importlib.util
import json
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("count_flips", _ROOT / "scripts" / "count_flips.py")
count_flips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(count_flips)

_CASES = {
    "pos": {"id": "pos", "expected_tool": "aggregate_sales",
            "expected_args": {"exact": {"period": "last_month"}, "semantic": {}}, "forbidden_tools": [], "negative": False},
    "neg": {"id": "neg", "expected_tool": None,
            "expected_args": {"exact": {}, "semantic": {}}, "forbidden_tools": [], "negative": True},
}
_RIGHT = {"tool": "aggregate_sales", "args": {"period": "last_month"}}
_WRONG_ARGS = {"tool": "aggregate_sales", "args": {"period": "this_month"}}
_WRONG_TOOL = {"tool": "trend_inventory", "args": {}}
_DECLINE = {"tool": None, "args": {}}


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def _row(case_id, trial, decoded, raw=None, call_id="a"):
    if raw is None:
        raw = json.dumps({"content": None, "_usage": {"prompt_tokens": trial},
                          "tool_calls": [{"id": call_id, "function": {"name": decoded["tool"]}}]})
    return {"case_id": case_id, "trial": trial, "decoded": decoded, "raw_output": raw}


def _run(tmp_path, base_rows, loaded_rows):
    base = count_flips.load_arm(_write(tmp_path / "c1.raw.jsonl", base_rows))
    loaded = count_flips.load_arm(_write(tmp_path / "c16.raw.jsonl", loaded_rows))
    return count_flips.count_flips(base, loaded, _CASES)


def test_identical_arms_have_no_flips_and_ids_or_usage_do_not_count_as_change(tmp_path):
    base = [_row("pos", t, _RIGHT, call_id="a") for t in range(2)]
    loaded = [_row("pos", t, _RIGHT, call_id="b") for t in range(2)]
    r = _run(tmp_path, base, loaded)
    assert r["flipped_cases"] == [] and r["raw_changed_cases"] == []
    assert r["n_cases"] == 1 and r["n_pairs"] == 2


def test_a_flip_is_a_selection_change_in_either_direction(tmp_path):
    base = [_row("pos", 0, _RIGHT), _row("pos", 1, _WRONG_TOOL), _row("neg", 0, _DECLINE)]
    loaded = [_row("pos", 0, _WRONG_TOOL), _row("pos", 1, _RIGHT), _row("neg", 0, _DECLINE)]
    r = _run(tmp_path, base, loaded)
    assert r["flipped_cases"] == ["pos"]
    assert (r["lost_under_load"], r["gained_under_load"]) == (1, 1)


def test_an_args_change_with_the_same_selection_is_not_a_flip(tmp_path):
    # ADR-0002's margin is on selection; a changed argument is reported, not counted.
    r = _run(tmp_path, [_row("pos", 0, _RIGHT)], [_row("pos", 0, _WRONG_ARGS)])
    assert r["flipped_cases"] == [] and r["changed_without_flip"] == ["pos"]


def test_reworded_decline_is_a_raw_change_only(tmp_path):
    base = [_row("neg", 0, _DECLINE, raw=json.dumps({"content": "はい。"}))]
    loaded = [_row("neg", 0, _DECLINE, raw=json.dumps({"content": "はい！"}))]
    r = _run(tmp_path, base, loaded)
    assert r["flipped_cases"] == [] and r["changed_without_flip"] == []
    assert r["raw_changed_cases"] == ["neg"]


def test_option_a_bound_matches_adr_0002(tmp_path):
    # ADR-0002: with 0 flipped cases, option A needs 141 cases for a 97.5% upper bound <= 2.6%.
    _, upper_141 = count_flips.clopper_pearson_one_sided(0, 141, 0.975)
    _, upper_140 = count_flips.clopper_pearson_one_sided(0, 140, 0.975)
    assert upper_141 <= 0.026 < upper_140


def test_arms_must_cover_the_same_pairs(tmp_path):
    with pytest.raises(ValueError, match="same"):
        _run(tmp_path, [_row("pos", 0, _RIGHT), _row("pos", 1, _RIGHT)], [_row("pos", 0, _RIGHT)])


def test_unrecovered_trial_is_refused(tmp_path):
    with pytest.raises(ValueError, match="trial"):
        count_flips.load_arm(_write(tmp_path / "x.raw.jsonl", [_row("pos", None, _RIGHT)]))
