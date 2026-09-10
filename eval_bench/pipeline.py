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
from typing import Any

from eval_bench.adapters.base import ModelAdapter
from eval_bench.dataset.tools import to_openai_tools_subset
from eval_bench.decoders.base import Decoder
from eval_bench.repetition import PredictFn
from eval_bench.scorers.separated import Prediction


def make_predict_fn(adapter: ModelAdapter, decoder: Decoder,
                     tools: dict[str, dict[str, Any]]) -> PredictFn:
    """Build a repetition.PredictFn backed by one adapter+decoder pair.

    One adapter instance should be reused across all trials of a cell (not
    reconstructed per trial): sampling_params -- including seed -- is fixed
    at adapter construction, and run_repeated_case's premise ("order changed
    between trials; nothing else did") requires that to stay constant too.
    """
    def predict(case: dict[str, Any], tool_names: list[str]) -> Prediction:
        tool_set = to_openai_tools_subset(tools, tool_names)
        raw = adapter.generate(case["query"], tool_set)
        return decoder.decode(raw, {})
    return predict
