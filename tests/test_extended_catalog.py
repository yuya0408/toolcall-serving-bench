"""Invariants for schema/tools_extended.json (the 100-tool catalog for
tool_stage=50/100, the engine-performance plan (kept outside this repo)). These are the checks a
visual read of the generated JSON won't catch: a duplicate name silently
collapses load_tools()'s dict to <100 entries, and stage=10 must still be
run against the ORIGINAL 40-tool file, never this one -- if a case can't be
validated or padded against the extended catalog at 50/100, that's a bug in
the catalog, not in the case.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval_bench.dataset.loader import load_cases
from eval_bench.dataset.tools import load_tools, validate_case_against_tools
from eval_bench.dataset.toolsets import build_tool_set

_ORIGINAL_PATH = pathlib.Path(__file__).resolve().parents[1] / "schema" / "tools.json"
_EXTENDED_PATH = pathlib.Path(__file__).resolve().parents[1] / "schema" / "tools_extended.json"


def test_extended_catalog_has_exactly_100_unique_tools():
    tools = load_tools(_EXTENDED_PATH)
    assert len(tools) == 100, "a duplicate name would silently collapse this below 100"


def test_original_40_are_carried_over_unchanged():
    original = load_tools(_ORIGINAL_PATH)
    extended = load_tools(_EXTENDED_PATH)
    assert len(original) == 40
    for name, orig_def in original.items():
        assert name in extended, f"{name} missing from extended catalog"
        assert extended[name] == orig_def, f"{name} drifted between tools.json and tools_extended.json"


def test_new_tools_are_erp_domain_not_disjoint():
    original = load_tools(_ORIGINAL_PATH)
    extended = load_tools(_EXTENDED_PATH)
    new_names = set(extended) - set(original)
    assert len(new_names) == 60
    modules = {extended[n]["module"] for n in new_names}
    # Every new tool must belong to a module that also appears in the
    # original 40 (deepening) or is one of the plan's two allowed additions.
    allowed = {t["module"] for t in original.values()} | {"production", "project"}
    assert modules <= allowed, f"unexpected module(s) introduced: {modules - allowed}"


def test_every_case_validates_against_extended_catalog():
    cases = load_cases("data")
    extended = load_tools(_EXTENDED_PATH)
    for case in cases:
        validate_case_against_tools(case, extended)


def test_build_tool_set_does_not_raise_at_stage_50_and_100():
    cases = load_cases("data")
    extended = load_tools(_EXTENDED_PATH)
    for case in cases:
        for size in (50, 100):
            tool_set = build_tool_set(case, extended, size)
            assert len(tool_set) == size
            if case.get("expected_tool") is not None:
                assert case["expected_tool"] in tool_set
            for d in case.get("distractor_tools", []):
                if d in extended:
                    assert d in tool_set


if __name__ == "__main__":
    test_extended_catalog_has_exactly_100_unique_tools()
    test_original_40_are_carried_over_unchanged()
    test_new_tools_are_erp_domain_not_disjoint()
    test_every_case_validates_against_extended_catalog()
    test_build_tool_set_does_not_raise_at_stage_50_and_100()
    print("extended catalog ok")
