"""The tools as SGLang re-serializes them (scripts/count_prompt_tokens.py)."""
from __future__ import annotations
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from count_prompt_tokens import sglang_tools  # noqa: E402


def test_sglang_tools_reorders_and_adds_strict_and_defer_loading():
    # SGLang 0.5.21's Tool(**t).model_dump(); checked equal on all 1,200 stage-10
    # requests on the instance (2026-10-28).
    sent = [{"type": "function", "function": {"name": "f", "description": "d",
                                              "parameters": {"type": "object"}}}]
    assert json.dumps(sglang_tools(sent)) == json.dumps(
        [{"type": "function", "function": {"description": "d", "name": "f", "parameters": {"type": "object"},
                                           "strict": False}, "defer_loading": None}])
