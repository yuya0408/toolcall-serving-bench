"""Wiring tests for the real ModelAdapter backends (eval_bench/adapters/).

No GPU / no live engine is involved: the HTTP adapter's transport is mocked
(unittest.mock) and the transformers_hf adapter's tokenizer/model are fakes
injected via the constructor (see transformers_hf.py's docstring on why the
constructor takes objects, not a model_id). These only check that each
adapter builds the request/call it should and hands back what its paired
Decoder (tests/test_decoders.py) expects -- actual engine connectivity is
exercised at GPU boot time, not here.
"""
from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

from eval_bench.adapters.openai_compatible import LlamaCppAdapter, VLLMAdapter
from eval_bench.adapters.transformers_hf import TransformersHFAdapter
from eval_bench.decoders.base import HermesTagDecoder, OpenAIToolCallDecoder

_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 256, "seed": 0}
_TOOL_SET = [{"type": "function", "function": {"name": "get_inventory", "parameters": {}}}]


def _fake_response(message: dict):
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": message}]}
    resp.raise_for_status.return_value = None
    return resp


def test_vllm_adapter_builds_request_and_returns_message_json():
    adapter = VLLMAdapter(base_url="http://localhost:8000", model="Qwen/Qwen3-8B",
                          sampling_params=_SAMPLING)
    message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_inventory", "arguments": json.dumps({"branch": "tokyo"})}},
    ]}

    with patch("requests.post", return_value=_fake_response(message)) as mock_post:
        raw = adapter.generate("在庫を確認して", _TOOL_SET)

    url, kwargs = mock_post.call_args[0][0], mock_post.call_args[1]
    assert url == "http://localhost:8000/v1/chat/completions"
    payload = kwargs["json"]
    assert payload["model"] == "Qwen/Qwen3-8B"
    assert payload["messages"] == [{"role": "user", "content": "在庫を確認して"}]
    assert payload["tools"] == _TOOL_SET
    assert payload["temperature"] == 0.7
    assert payload["seed"] == 0

    # Round-trips cleanly through the paired decoder.
    pred = OpenAIToolCallDecoder().decode(raw, {})
    assert pred.tool == "get_inventory"
    assert pred.args == {"branch": "tokyo"}


def test_llama_cpp_adapter_shares_the_same_request_shape():
    adapter = LlamaCppAdapter(base_url="http://localhost:8080", model="qwen3-8b-bf16.gguf",
                              sampling_params=_SAMPLING)
    message = {"role": "assistant", "content": "ツールは使いません。", "tool_calls": None}

    with patch("requests.post", return_value=_fake_response(message)) as mock_post:
        raw = adapter.generate("今日は何曜日ですか", _TOOL_SET)

    assert mock_post.call_args[0][0] == "http://localhost:8080/v1/chat/completions"
    assert OpenAIToolCallDecoder().decode(raw, {}) == \
        OpenAIToolCallDecoder().decode(json.dumps(message), {})


def test_vllm_and_llama_cpp_adapters_differ_only_in_name_and_target():
    assert VLLMAdapter.name == "vllm"
    assert LlamaCppAdapter.name == "llama_cpp"
    assert VLLMAdapter.__mro__[1] is LlamaCppAdapter.__mro__[1]  # same shared base


class _FakeTensor(list):
    @property
    def shape(self):
        return (1, len(self))

    def to(self, device):
        return self


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self, generated_text: str):
        self.generated_text = generated_text
        self.last_apply_chat_template_kwargs = None
        self.last_prompt = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        self.last_apply_chat_template_kwargs = kwargs
        return "PROMPT"

    def __call__(self, prompt, return_tensors="pt"):
        self.last_prompt = prompt
        return _FakeBatch(input_ids=_FakeTensor([1, 2, 3]))  # pretend a 3-token prompt

    def decode(self, ids, skip_special_tokens=True):
        return self.generated_text


class _FakeModel:
    device = "cpu"

    def __init__(self, generated_ids: list[int]):
        self.generated_ids = generated_ids
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return [[1, 2, 3] + self.generated_ids]  # prompt echoed back + new tokens, as HF does


def test_transformers_hf_adapter_uses_chat_template_with_tools_and_no_thinking():
    # seed=None here: setting a manual seed needs torch (see generate()'s seed
    # branch), which this no-GPU environment doesn't have installed. That
    # branch is exercised at GPU boot, not by this wiring test.
    tokenizer = _FakeTokenizer(generated_text="<tool_call>{\"name\": \"get_inventory\", "
                                               "\"arguments\": {\"branch\": \"tokyo\"}}</tool_call>")
    model = _FakeModel(generated_ids=[99, 99, 99])
    adapter = TransformersHFAdapter(tokenizer, model, {**_SAMPLING, "seed": None})

    raw = adapter.generate("在庫を確認して", _TOOL_SET)

    assert tokenizer.last_apply_chat_template_kwargs["tools"] == _TOOL_SET
    assert tokenizer.last_apply_chat_template_kwargs["enable_thinking"] is False
    assert model.last_kwargs["max_new_tokens"] == 256
    assert model.last_kwargs["do_sample"] is True  # temperature=0.7 > 0

    pred = HermesTagDecoder().decode(raw, {})
    assert pred.tool == "get_inventory"
    assert pred.args == {"branch": "tokyo"}


def test_transformers_hf_adapter_greedy_when_temperature_zero():
    tokenizer = _FakeTokenizer(generated_text="ツールは使いません。")
    model = _FakeModel(generated_ids=[1])
    adapter = TransformersHFAdapter(tokenizer, model, {**_SAMPLING, "temperature": 0.0, "seed": None})

    adapter.generate("今日は何曜日ですか", _TOOL_SET)

    assert model.last_kwargs["do_sample"] is False
