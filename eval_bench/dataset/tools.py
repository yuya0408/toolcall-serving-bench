"""Tool definitions are the single source of truth.

Two things are derived from schema/tools.json rather than restated by hand:

  1. Which argument fields are scored exact vs semantic (x-scoring).
  2. Whether a case's expected_args are legal for the tool it names.

This prevents the failure mode where a case file and the tool definition drift
apart and the harness silently scores against a stale contract.
"""
from __future__ import annotations
import json, pathlib
from typing import Any

_TOOLS_PATH = pathlib.Path(__file__).resolve().parents[2] / "schema" / "tools.json"


def load_tools(path: str | pathlib.Path = _TOOLS_PATH) -> dict[str, dict[str, Any]]:
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return {t["name"]: t for t in doc["tools"]}


def scoring_split(tool: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return (exact_fields, semantic_fields) for a tool, from x-scoring."""
    props = tool["parameters"]["properties"]
    exact = {k for k, v in props.items() if v.get("x-scoring") != "semantic"}
    semantic = {k for k, v in props.items() if v.get("x-scoring") == "semantic"}
    return exact, semantic


def _to_openai_tool(t: dict[str, Any]) -> dict[str, Any]:
    """Render one tool definition in the OpenAI 'tools' array entry shape.
    x-scoring is stripped: it is harness metadata, not something the model sees.
    """
    props = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("x-")}
        for k, v in t["parameters"]["properties"].items()
    }
    return {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": {**t["parameters"], "properties": props},
        },
    }


def to_openai_tools(tools: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Render every tool in `tools`, in the dict's own (catalog) order.

    Only useful when the whole catalog should be presented as-is. A trial
    that presents a stage-sized, shuffled subset (eval_bench/repetition.py's
    build_tool_set output) must use to_openai_tools_subset instead --
    iterating tools.values() here would silently discard both the subset
    and the shuffle, which is the entire point of that experiment.
    """
    return [_to_openai_tool(t) for t in tools.values()]


def to_openai_tools_subset(tools: dict[str, dict[str, Any]],
                           names: list[str]) -> list[dict[str, Any]]:
    """Render only `names`, in that exact order.

    This is the variant a trial must use: build_tool_set returns a shuffled,
    stage-sized subset of tool names, and the model-facing tool array has to
    preserve that order (it's what run_repeated_case's consistency measure
    and the prompt_layout/prefix-caching axis both depend on) and that
    subset (an unfiltered to_openai_tools call would leak every tool in the
    catalog into every trial, defeating the tool-count experiment).
    """
    return [_to_openai_tool(tools[name]) for name in names]


def validate_case_against_tools(case: dict[str, Any], tools: dict[str, dict[str, Any]]) -> None:
    """Raise if a case references an unknown tool or illegal/misclassified args."""
    name = case.get("expected_tool")
    if name is None:
        if not case.get("negative"):
            raise ValueError(f"{case['id']}: expected_tool is null but negative is false")
        return
    if name not in tools:
        raise ValueError(f"{case['id']}: unknown tool {name!r}")

    exact_fields, semantic_fields = scoring_split(tools[name])
    args = case.get("expected_args", {})

    for k in args.get("exact", {}):
        if k not in exact_fields:
            raise ValueError(
                f"{case['id']}: {k!r} is not an exact-scored field of {name!r}"
            )
    for k in args.get("semantic", {}):
        if k not in semantic_fields:
            raise ValueError(
                f"{case['id']}: {k!r} is not a semantic-scored field of {name!r}"
            )

    # enum values in the case must be legal per the tool definition
    props = tools[name]["parameters"]["properties"]
    for k, v in args.get("exact", {}).items():
        allowed = props[k].get("enum")
        if allowed is not None and v not in allowed:
            raise ValueError(f"{case['id']}: {k}={v!r} not in enum {allowed}")

    for k in tools[name]["parameters"].get("required", []):
        if k not in args.get("exact", {}) and k not in args.get("semantic", {}):
            raise ValueError(f"{case['id']}: required arg {k!r} missing from expected_args")
