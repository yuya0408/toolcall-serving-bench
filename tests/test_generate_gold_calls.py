"""scripts/generate_gold_calls.py: the gold calls behind the generated cases
(docs/adr/0004-expand-case-set.md) must be reproducible, distinct, legal
per the tool catalog, and free of the combinations no user would ask for."""
from __future__ import annotations
import importlib.util
import json
import pathlib
import random
import sys
from collections import Counter

import pytest

from eval_bench.dataset.loader import load_cases
from eval_bench.dataset.tools import load_tools

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("generate_gold_calls", _ROOT / "scripts" / "generate_gold_calls.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

_TOOLS = load_tools(_ROOT / "schema" / "tools.json")
_EXISTING = [c for c in load_cases(_ROOT / "data") if c["provenance"] != "generated"]


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    out = tmp_path_factory.mktemp("gold") / "gold_calls.jsonl"
    assert gen.main(["--out", str(out)]) == 0
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def test_reproducible(rows, tmp_path):
    out = tmp_path / "again.jsonl"
    gen.main(["--out", str(out)])
    assert [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] == rows


def test_counts_fill_200_with_30_negatives_and_4_per_tool(rows):
    existing_pos = [c for c in _EXISTING if not c["negative"]]
    assert len(rows) + len(existing_pos) == gen.TOTAL - gen.NEGATIVES
    per_tool = Counter(r["expected_tool"] for r in rows) + Counter(c["expected_tool"] for c in existing_pos)
    assert set(per_tool) == set(_TOOLS)
    assert min(per_tool.values()) >= gen.MIN_PER_TOOL


def test_every_gold_call_is_distinct_including_existing(rows):
    keys = [gen._key(r["expected_tool"], r["expected_args"]["exact"]) for r in rows]
    keys += [gen._key(c["expected_tool"], c["expected_args"]["exact"]) for c in _EXISTING if not c["negative"]]
    assert len(keys) == len(set(keys))


def test_required_present_sentinels_absent_and_plausible(rows):
    for r in rows:
        tool = _TOOLS[r["expected_tool"]]
        args = r["expected_args"]["exact"]
        assert set(tool["parameters"].get("required", [])) <= args.keys(), r["id"]
        assert not gen.SENTINELS & {v for v in args.values() if isinstance(v, str)}, r["id"]
        assert gen.plausible(tool, args), r["id"]
        if {"year_from", "year_to"} <= args.keys():
            assert args["year_from"] < args["year_to"]


def test_distractors_are_other_real_tools(rows):
    for r in rows:
        assert r["expected_tool"] not in r["distractor_tools"]
        assert set(r["distractor_tools"]) <= set(_TOOLS)
        assert 1 <= len(r["distractor_tools"]) <= 3


@pytest.mark.parametrize("tool,args", [
    ("aggregate_hr", {"entity": "headcount", "aggregation": "sum"}),                  # summing people
    ("aggregate_sales", {"entity": "invoice", "aggregation": "sum", "group_by": "month", "period": "last_month"}),
    ("search_master", {"entity": "employee", "field": "credit_limit"}),
    ("compare_period_metric", {"metric": "sales", "period_a": "last_fiscal_year", "period_b": "last_month"}),
    ("trend_sales", {"entity": "invoice", "metric": "amount", "interval": "quarter", "period": "this_month"}),
    ("search_report", {"report_type": "yuho"}),
    ("search_report_section", {"report_type": "ordinance"}),
])
def test_implausible_combinations_are_rejected(tool, args):
    assert not gen.plausible(_TOOLS[tool], args)


def test_document_numbers_follow_the_document_type():
    rng = random.Random(0)
    tool = _TOOLS["get_purchase_detail"]
    spec = tool["parameters"]["properties"]["document_no"]
    assert gen._value(rng, tool, "document_no", spec, {"entity": "receipt"}).startswith("RC-")


def test_flow_aggregates_and_trends_have_a_period(rows):
    missing = [r["id"] for r in rows if gen.needs_period(_TOOLS[r["expected_tool"]], r["expected_args"]["exact"])]
    assert missing == []


def test_no_open_items_over_snapshot_entities(rows):
    bad = [r["id"] for r in rows if gen.snapshot_open_items(_TOOLS[r["expected_tool"]], r["expected_args"]["exact"])]
    assert bad == []
