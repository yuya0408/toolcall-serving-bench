"""Glue between eval_bench/repetition.py and a real (ModelAdapter, Decoder)
pair. repetition.py deliberately knows nothing about backends -- its
PredictFn is `(case, tool_names) -> Prediction`, where tool_names is
build_tool_set's shuffled, stage-sized subset of tool names.

Turning that back into a model-facing tool array is the one step that must
not use dataset/tools.py's to_openai_tools(): that function iterates the
whole catalog in its own (unshuffled) order, which would silently discard
both the stage's subset and run_repeated_case's shuffle -- the two things
the tool-count and consistency experiments depend on. to_openai_tools_subset
is the ordered-subset variant that preserves both.
"""
from __future__ import annotations
import json
import re
import time
from typing import Any, Callable, Optional

from eval_bench.adapters.base import ModelAdapter
from eval_bench.dataset.tools import to_openai_tools_subset
from eval_bench.decoders.base import Decoder
from eval_bench.repetition import PredictFn
from eval_bench.scorers.separated import Prediction


# (case, tool_names, raw_output, decoded_prediction, elapsed_seconds)
CallRecorder = Callable[[dict[str, Any], list[str], str, Prediction, float], None]

# A non-empty <think> block. Qwen3 with enable_thinking=False may still emit an
# EMPTY <think></think> pair (whitespace only inside), which is harmless; only
# content inside it means the model actually reasoned.
_NONEMPTY_THINK_RE = re.compile(r"<think>\s*\S.*?</think>", re.DOTALL)


def assert_no_thinking(raw: str) -> None:
    """Fail the cell if the model produced reasoning despite
    enable_thinking=False.

    tests/test_adapters.py proves every adapter *sends* thinking-off; this is
    the runtime half that proves the engine *honored* it (a server can
    ignore chat_template_kwargs, e.g. a llama.cpp build without --jinja
    support for it). A silently-thinking engine would fold reasoning-trace
    quality and its token cost into the "engine delta" -- the same silent
    comparability failure the adapter parity test exists to prevent, so it
    raises rather than logs.

    Handles both raw shapes: the OpenAI-compatible adapters' JSON-serialized
    `message` (reasoning in `reasoning_content`, or inline in `content`) and
    transformers_hf's plain decoded text.
    """
    texts = [raw]
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        message = None
    if isinstance(message, dict):
        if (message.get("reasoning_content") or "").strip():
            raise ValueError("engine returned non-empty reasoning_content with thinking disabled")
        texts = [message.get("content") or ""]
    for text in texts:
        if _NONEMPTY_THINK_RE.search(text):
            raise ValueError("engine emitted a non-empty <think> block with thinking disabled")


def make_predict_fn(adapter: ModelAdapter, decoder: Decoder,
                     tools: dict[str, dict[str, Any]],
                     recorder: Optional[CallRecorder] = None) -> PredictFn:
    """Build a repetition.PredictFn backed by one adapter+decoder pair.

    One adapter instance should be reused across all trials of a cell (not
    reconstructed per trial): sampling_params -- including seed -- is fixed
    at adapter construction, and run_repeated_case's premise ("order changed
    between trials; nothing else did") requires that to stay constant too.

    `recorder`, if given, sees every call's raw output (before decoding) --
    scripts/run_cell.py uses it to keep a raw-output log, so a cell can be
    rescored offline after the GPU instance is gone. It may be called from
    several threads at once (run_repeated_cases_concurrent).
    """
    def predict(case: dict[str, Any], tool_names: list[str]) -> Prediction:
        tool_set = to_openai_tools_subset(tools, tool_names)
        started = time.perf_counter()
        raw = adapter.generate(case["query"], tool_set)
        elapsed = time.perf_counter() - started
        assert_no_thinking(raw)
        pred = decoder.decode(raw, {})
        if recorder is not None:
            recorder(case, tool_names, raw, pred, elapsed)
        return pred
    return predict
