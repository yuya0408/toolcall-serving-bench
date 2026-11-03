"""Golden checks for the CI regression gate (docs/adr/0010-ci-golden-gate.md).

The gate never runs a model. It replays the 8B answers saved from the main
runs (one bf16 accuracy cell per engine, tests/golden/) through the current
harness code and compares, in the order a request flows:

  G1 tool order   trial_order() rebuilt per (case, trial)    vs the recorded tool_order
  G2 request      the body each engine would be sent          vs the expected file
  G3 decoding     the decoder applied to the saved raw_output vs the recorded decoded
  G4 scoring      score_case() on the recorded decoded        vs the expected file

G1 and G3 compare against the run records themselves; G2 and G4 compare
against files written from main by scripts/update_golden.py. Each check
returns its mismatches so a test can assert none and a deliberately broken
harness can be shown to produce some.
"""
from __future__ import annotations
import gzip
import hashlib
import io
import json
import pathlib
import sys
import types
from typing import Any, Callable, Iterable
from unittest.mock import patch

from eval_bench.dataset.loader import load_cases
from eval_bench.dataset.tools import load_tools, to_openai_tools_subset
from eval_bench.dataset.toolsets import build_tool_set
from eval_bench.repetition import trial_order
from eval_bench.scorers.separated import Prediction, score_case

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden"
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# The cells the gate replays: one bf16 plain-accuracy cell per engine, from
# the main runs in Release runs-2026-10 (docs/adr/0010 5).
CELLS = {
    "transformers_hf": "gpu-20261025-i-062c245383e4af847/acc-hf-bf16",
    "llama_cpp": "gpu-20261025-i-062c245383e4af847/acc-llama-gguf",
    "vllm": "gpu-20261025-i-062c245383e4af847/acc-vllm-bf16",
    "sglang": "gpu-20261028-i-0bd6a9b5da22d5bf5/acc-sglang-bf16",
    "tensorrt_llm": "gpu-20261028-i-0bd6a9b5da22d5bf5/acc-trt-bf16",
}
ANSWER_FIELDS = ("case_id", "trial", "tool_order", "raw_output", "decoded")


def _no_semantic_judge(expected: str, predicted: str) -> bool:
    raise RuntimeError("a semantic-scored field was scored, but the catalog has none")


# --- files -----------------------------------------------------------------

def write_jsonl_gz(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    """Byte-stable: mtime=0 and LF, so regenerating unchanged rows changes nothing."""
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as f:
        f.write(text.encode("utf-8"))
    path.write_bytes(buf.getvalue())


def read_jsonl_gz(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()]


def load_golden(engine: str, golden_dir: pathlib.Path = GOLDEN_DIR) -> tuple[dict, list[dict], list[dict]]:
    """(cell, answers, expected) for one engine."""
    cells = json.loads((golden_dir / "cells.json").read_text(encoding="utf-8"))
    return (cells[engine], read_jsonl_gz(golden_dir / f"{engine}.answers.jsonl.gz"),
            read_jsonl_gz(golden_dir / f"{engine}.expected.jsonl.gz"))


def harness_inputs(data_dir: pathlib.Path = DATA_DIR) -> tuple[dict[str, dict], dict[str, dict]]:
    return {c["id"]: c for c in load_cases(data_dir)}, load_tools()


# --- what the harness computes ------------------------------------------------

def tool_order(case: dict, tools: dict, cell: dict, trial: int) -> list[str]:
    tool_set = build_tool_set(case, tools, cell["tool_stage"], seed=0)
    return trial_order(case["id"], tool_set, cell["tool_stage"], trial, cell["seed_base"])


def decoder_for(engine: str):
    from eval_bench.decoders.base import HermesTagDecoder, OpenAIToolCallDecoder
    return HermesTagDecoder() if engine == "transformers_hf" else OpenAIToolCallDecoder()


def _hf_request(cell: dict, query: str, tool_set: list[dict]) -> dict[str, Any]:
    """What TransformersHFAdapter hands to the chat template and generate():
    the template itself runs in transformers (pinned, docs/adr/0006), so the
    gate checks its inputs, with the tokenizer, model and torch faked."""
    from eval_bench.adapters.transformers_hf import TransformersHFAdapter

    seen: dict[str, Any] = {}

    class _Batch(dict):
        def to(self, device):
            return self

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"], seen["template_kwargs"] = messages, kwargs
            return "PROMPT"

        def __call__(self, prompt, return_tensors="pt"):
            return _Batch(input_ids=None)

    model = types.SimpleNamespace(device="cpu")
    fake_torch = types.SimpleNamespace(manual_seed=lambda s: seen.__setitem__("seed", s))
    with patch.dict(sys.modules, {"torch": fake_torch}):
        _, gen_kwargs = TransformersHFAdapter(_Tokenizer(), model, cell["sampling_params"])._prepare(query, tool_set)
    kwargs = dict(seen["template_kwargs"])
    tools = kwargs.pop("tools")
    return {"messages": seen["messages"], "tools": tools, "template_kwargs": kwargs,
            "gen_kwargs": gen_kwargs, "seed": seen.get("seed")}


def request(engine: str, cell: dict, case: dict, tools: dict, order: list[str]) -> dict[str, Any]:
    tool_set = to_openai_tools_subset(tools, order)
    if engine == "transformers_hf":
        return _hf_request(cell, case["query"], tool_set)
    from eval_bench.adapters import openai_compatible as oc
    cls = {"vllm": oc.VLLMAdapter, "sglang": oc.SGLangAdapter, "tensorrt_llm": oc.TensorRTLLMAdapter,
           "llama_cpp": oc.LlamaCppAdapter}[engine]
    return cls(base_url="http://golden", model=cell["model"],
               sampling_params=cell["sampling_params"]).build_payload(case["query"], tool_set)


def request_record(req: dict[str, Any]) -> dict[str, Any]:
    """The tool definitions repeat in every row, so the expected file keeps
    their names readable and the full body as a digest."""
    body = json.dumps(req, ensure_ascii=False, sort_keys=True).encode("utf-8")
    summary = {k: v for k, v in req.items() if k != "tools"}
    summary["tool_names"] = [t["function"]["name"] for t in req["tools"]]
    return {"summary": summary, "sha256": hashlib.sha256(body).hexdigest()}


def score(case: dict, decoded: dict) -> dict[str, Any]:
    s = score_case(case, Prediction(decoded["tool"], decoded["args"]), _no_semantic_judge)
    return {"selection": s.selection, "arg_exact": s.arg_exact,
            "arg_semantic": s.arg_semantic, "e2e": s.e2e}


def expected_rows(engine: str, cell: dict, answers: list[dict], cases: dict, tools: dict) -> list[dict]:
    return [{"case_id": a["case_id"], "trial": a["trial"],
             "request": request_record(request(engine, cell, cases[a["case_id"]], tools, a["tool_order"])),
             "score": score(cases[a["case_id"]], a["decoded"])} for a in answers]


# --- the four checks -----------------------------------------------------------

Mismatch = tuple[str, int, Any, Any]  # (case_id, trial, expected, got)


def _diff(rows: Iterable[tuple[dict, Any, Any]]) -> list[Mismatch]:
    return [(r["case_id"], r["trial"], want, got) for r, want, got in rows if want != got]


def check_g1_tool_order(cell: dict, answers: list[dict], cases: dict, tools: dict,
                        order_fn: Callable = tool_order) -> list[Mismatch]:
    return _diff((a, a["tool_order"], order_fn(cases[a["case_id"]], tools, cell, a["trial"])) for a in answers)


def check_g2_request(engine: str, cell: dict, answers: list[dict], expected: list[dict], cases: dict,
                     tools: dict, request_fn: Callable = request) -> list[Mismatch]:
    return _diff((a, e["request"], request_record(request_fn(engine, cell, cases[a["case_id"]], tools, a["tool_order"])))
                 for a, e in zip(answers, expected))


def check_g3_decoding(engine: str, answers: list[dict], decoder=None) -> list[Mismatch]:
    decoder = decoder or decoder_for(engine)

    def decoded(a):
        p = decoder.decode(a["raw_output"], {})
        return {"tool": p.tool, "args": p.args}
    return _diff((a, a["decoded"], decoded(a)) for a in answers)


def check_g4_scoring(answers: list[dict], expected: list[dict], cases: dict,
                     score_fn: Callable = score) -> list[Mismatch]:
    return _diff((a, e["score"], score_fn(cases[a["case_id"]], a["decoded"])) for a, e in zip(answers, expected))


def selection_mean(expected: list[dict]) -> float:
    return sum(e["score"]["selection"] for e in expected) / len(expected)
