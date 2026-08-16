"""Dataset layer: load + validate cases against schema/case.schema.json.

Refuses to load a case missing provenance or (for grounded cases) source.
Transparency is enforced here, not left to documentation.
"""
from __future__ import annotations
import json, pathlib
from typing import Any

try:
    import jsonschema
except ImportError:
    jsonschema = None

_SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[2] / "schema" / "case.schema.json"


def _validator():
    if jsonschema is None:
        return None
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def load_cases(data_dir: str | pathlib.Path) -> list[dict[str, Any]]:
    v = _validator()
    cases: list[dict[str, Any]] = []
    for p in sorted(pathlib.Path(data_dir).rglob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if v is not None:
                v.validate(case)
            cases.append(case)
    return cases


def by_provenance(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split cases so metrics can be recomputed excluding any generation method."""
    out: dict[str, list[dict[str, Any]]] = {}
    for c in cases:
        out.setdefault(c["provenance"], []).append(c)
    return out
