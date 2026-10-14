"""Tests for eval_bench/pipeline.py's PredictFn glue.

The one bug this guards against (flagged during review): dataset/tools.py's
to_openai_tools() iterates the whole catalog in its own order, discarding
both build_tool_set's shuffle and its stage-sized subset if used here by
mistake. If that regressed, every trial would send an identical, full tool
list and run_repeated_case's consistency measure would become meaningless
without raising any error -- so this is checked directly against a mocked
adapter's captured request, not just against to_openai_tools_subset in
isolation (tests/test_tools.py doesn't exist yet, but this covers the
integration point where the bug would actually bite).
"""
from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

from eval_bench.adapters.base import EchoAdapter
from eval_bench.adapters.openai_compatible import VLLMAdapter
from eval_bench.decoders.base import IdentityDecoder, OpenAIToolCallDecoder
from eval_bench.pipeline import make_predict_fn

_TOOLS = {
    "get_inventory": {
        "name": "get_inventory", "description": "在庫を取得",
        "parameters": {"type": "object", "properties": {"branch": {"type": "string"}}, "required": ["branch"]},
    },
    "get_sales": {
        "name": "get_sales", "description": "売上を取得",
        "parameters": {"type": "object", "properties": {"period": {"type": "string"}}, "required": []},
    },
    "get_purchase": {
        "name": "get_purchase", "description": "購買を取得",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "get_hr": {
        "name": "get_hr", "description": "人事情報を取得",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
_CASE = {"id": "case_x", "query": "在庫を確認して"}


def _fake_response(message: dict):
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": message}]}
    resp.raise_for_status.return_value = None
    return resp


def test_predict_fn_preserves_shuffled_subset_order_in_the_actual_request():
    # Deliberately not catalog order, and deliberately a subset (3 of 4).
    shuffled_subset = ["get_purchase", "get_inventory", "get_hr"]
    adapter = VLLMAdapter(base_url="http://localhost:8000", model="Qwen/Qwen3-8B",
                          sampling_params={"temperature": 0.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0, "max_tokens": 64, "seed": 0})
    predict = make_predict_fn(adapter, OpenAIToolCallDecoder(), _TOOLS)

    message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "get_inventory", "arguments": json.dumps({"branch": "tokyo"})}},
    ]}
    with patch("requests.post", return_value=_fake_response(message)) as mock_post:
        pred = predict(_CASE, shuffled_subset)

    sent_tool_names = [t["function"]["name"] for t in mock_post.call_args[1]["json"]["tools"]]
    assert sent_tool_names == shuffled_subset  # order preserved, not catalog order
    assert len(sent_tool_names) == 3           # subset, not the full 4-tool catalog
    assert pred.tool == "get_inventory"
    assert pred.args == {"branch": "tokyo"}


def test_predict_fn_works_with_the_echo_stub_for_backend_free_testing():
    """README/repetition.py advertise stub-testability as a feature -- this
    is what that looks like wired through the same glue real backends use."""
    predict = make_predict_fn(EchoAdapter(), IdentityDecoder(), _TOOLS)
    pred = predict(_CASE, ["get_inventory", "get_sales"])
    assert pred.tool is None  # EchoAdapter always returns "{}"
