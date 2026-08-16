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


def to_openai_tools(tools: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Render tool definitions in the common 'tools' array shape for adapters.
    x-scoring is stripped: it is harness metadata, not something the model sees.
    """
    out = []
    for t in tools.values():
        props = {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("x-")}
            for k, v in t["parameters"]["properties"].items()
        }
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {**t["parameters"], "properties": props},
            },
        })
    return out


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
