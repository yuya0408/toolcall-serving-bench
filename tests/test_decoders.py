"""Tests for the engine-specific Decoders added for the chat/tools API switch
(docs/engine-performance-plan.md's '比較可能性の規律'). Each covers the same
three cases: a normal tool call, a well-formed decline (no call at all), and
a malformed/partial output -- and checks that a parse failure never gets
mistaken for a selection failure or vice versa (scorers/separated.py's
separated-scoring guarantee).
"""
import json

from eval_bench.decoders.base import HermesTagDecoder, OpenAIToolCallDecoder
from eval_bench.scorers.separated import Prediction


def test_hermes_decoder_parses_tool_call():
    dec = HermesTagDecoder()
    raw = (
        "<tool_call>\n"
        '{"name": "get_inventory", "arguments": {"branch": "tokyo"}}\n'
        "</tool_call>"
    )
    pred = dec.decode(raw, {})
    assert pred == Prediction(tool="get_inventory", args={"branch": "tokyo"})


def test_hermes_decoder_no_tag_is_a_decline():
    dec = HermesTagDecoder()
    pred = dec.decode("今日は何曜日ですか、というご質問にはツールは使えません。", {})
    assert pred == Prediction(tool=None, args={})


def test_hermes_decoder_malformed_inner_json_is_a_parse_failure():
    dec = HermesTagDecoder()
    raw = "<tool_call>\n{not valid json\n</tool_call>"
    pred = dec.decode(raw, {})
    assert pred == Prediction(tool=None, args={})


def test_hermes_decoder_reads_only_first_of_multiple_calls():
    dec = HermesTagDecoder()
    raw = (
        "<tool_call>{\"name\": \"first_tool\", \"arguments\": {}}</tool_call>"
        "<tool_call>{\"name\": \"second_tool\", \"arguments\": {}}</tool_call>"
    )
    pred = dec.decode(raw, {})
    assert pred.tool == "first_tool"


def _openai_message(tool_calls=None, content=None):
    return json.dumps({"role": "assistant", "content": content, "tool_calls": tool_calls})


def test_openai_decoder_parses_tool_call():
    dec = OpenAIToolCallDecoder()
    raw = _openai_message(tool_calls=[{
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_inventory", "arguments": json.dumps({"branch": "tokyo"})},
    }])
    pred = dec.decode(raw, {})
    assert pred == Prediction(tool="get_inventory", args={"branch": "tokyo"})


def test_openai_decoder_no_tool_calls_is_a_decline():
    dec = OpenAIToolCallDecoder()
    raw = _openai_message(tool_calls=None, content="ツールは使いません。")
    pred = dec.decode(raw, {})
    assert pred == Prediction(tool=None, args={})


def test_openai_decoder_malformed_arguments_keeps_tool_name():
    """A parse failure inside `arguments` is an arg-extraction failure, not a
    selection failure -- selection and arg metrics must never be conflated
    (scorers/separated.py's separated-scoring design)."""
    dec = OpenAIToolCallDecoder()
    raw = _openai_message(tool_calls=[{
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_inventory", "arguments": "{not valid json"},
    }])
    pred = dec.decode(raw, {})
    assert pred.tool == "get_inventory"
    assert pred.args == {}
