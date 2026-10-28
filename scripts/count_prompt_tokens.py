#!/usr/bin/env python3
"""Count the longest prompt a cell will send, without a GPU (ADR-0005 2).

Builds every request exactly as scripts/run_cell.py does (build_tool_set,
trial_order, to_openai_tools_subset; latency queries carry the request-id
suffix) and renders it with Qwen3-8B's chat template at a pinned revision,
the way the servers do: apply_chat_template(..., tools=...,
add_generation_prompt=True, enable_thinking=False).

The 2026-10-18 smoke run checked this count against vLLM's prompt_tokens: equal
on all 90 latency requests (+-1 from the random request id). llama.cpp reported
4 tokens fewer, but that was the request id, not the engine: the ids were random
hex, whose token count varies, and the two cells drew different ids. llama.cpp
renders the same template as HF token for token (checked 2026-10-20). The ids
are now digits, whose token count does not depend on the value. The capacity
check still adds one allocation unit on top, for differences not checked.

Needs `transformers` (tokenizer only; no torch) and the tokenizer files, which
it downloads from the Hub on first use:

    python scripts/count_prompt_tokens.py --tool-stage 10 --write-budget

--write-budget records the result in schema/prompt_budget.json, which the GPU
runs read (eval_bench/capacity.py). --engine sglang counts what SGLang renders
instead (sglang_tools) and records it under the entry's by_engine. The entry carries the catalog version and a
digest of the cases, so editing either makes the runs refuse the stale count.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from eval_bench.capacity import PROMPT_BUDGET_PATH, cases_digest  # noqa: E402
from eval_bench.dataset.loader import load_cases  # noqa: E402
from eval_bench.dataset.tools import load_tools, to_openai_tools_subset  # noqa: E402
from eval_bench.dataset.toolsets import build_tool_set  # noqa: E402
from eval_bench.latency import tagged_query  # noqa: E402
from eval_bench.repetition import trial_order  # noqa: E402

# The revision the GPU runs pinned (runs/gpu-20261018/versions.json). The bf16
# and AWQ repos ship the same chat template at the pinned revisions.
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
_CATALOGS = {10: "tools.json", 50: "tools_extended.json", 100: "tools_extended.json"}
# The latency pass's request ids are random digits (eval_bench.latency.new_run_tag);
# every value costs the same number of tokens, so one fixed id stands for all.
_TAG = "0" * 8


def sglang_tools(tool_set: list[dict]) -> list[dict]:
    """The tools as SGLang 0.5.21 hands them to the chat template: it parses
    each into its Tool model and dumps it back (serving_chat.py,
    item.model_dump()), which reorders the function's keys and adds
    "strict": false and "defer_loading": null (protocol.py, Function and Tool).
    About 10 tokens per tool; the smoke apply (2026-10-28) saw +100 on every
    case at stage 10 against vLLM and TensorRT-LLM (docs/engines.md)."""
    return [{"type": t.get("type", "function"),
             "function": {"description": t["function"].get("description"), "name": t["function"]["name"],
                          "parameters": t["function"].get("parameters"),
                          "strict": t["function"].get("strict", False)},
             "defer_loading": t.get("defer_loading")}
            for t in tool_set]


_SERIALIZE = {"default": lambda tool_set: tool_set, "sglang": sglang_tools}


def _count(tokenizer, query: str, tool_set: list[dict]) -> int:
    # Render to text first, then tokenize, as the servers do (the template
    # already contains every special token, so none are added again).
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}], tools=tool_set,
        add_generation_prompt=True, enable_thinking=False, tokenize=False,
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def count(cases, tools, stage: int, tokenizer, trials: int, engine: str = "default") -> dict:
    """Per-case maximum over every request the cell can send for that case:
    the accuracy passes' `trials` shuffled orders and the latency pass's fixed
    seed-0 order with a request id appended."""
    serialize = _SERIALIZE[engine]
    per_case = {}
    for case in cases:
        tool_set = build_tool_set(case, tools, stage, seed=0)
        lengths = [_count(tokenizer, case["query"],
                          serialize(to_openai_tools_subset(tools, trial_order(case["id"], tool_set, stage, t))))
                   for t in range(trials)]
        fixed = serialize(to_openai_tools_subset(tools, tool_set))
        lengths.append(_count(tokenizer, tagged_query(case["query"], _TAG, 999_999), fixed))
        per_case[case["id"]] = max(lengths)
    longest = max(per_case, key=per_case.get)
    return {
        "tool_stage": stage,
        "cases": len(per_case),
        "max_prompt_tokens": per_case[longest],
        "max_case": longest,
        "median_prompt_tokens": statistics.median(per_case.values()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tool-stage", type=int, required=True, choices=sorted(_CATALOGS))
    p.add_argument("--data-dir", default=str(_ROOT / "data"))
    p.add_argument("--trials", type=int, default=5, help="shuffled orders per case (the accuracy passes' n)")
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--engine", default="default", choices=sorted(_SERIALIZE),
                   help="sglang: the tools as SGLang re-serializes them (sglang_tools)")
    p.add_argument("--write-budget", action="store_true",
                   help=f"record the result in {PROMPT_BUDGET_PATH.relative_to(_ROOT)}")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer  # imported late: only this script needs it
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision=args.revision)
    tools = load_tools(_ROOT / "schema" / _CATALOGS[args.tool_stage])
    cases = load_cases(args.data_dir)
    result = count(cases, tools, args.tool_stage, tokenizer, args.trials, args.engine)
    result["tokenizer_revision"] = args.revision
    result["tool_catalog_version"] = json.loads(
        (_ROOT / "schema" / _CATALOGS[args.tool_stage]).read_text(encoding="utf-8"))["version"]
    result["cases_sha256"] = cases_digest(cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.write_budget:
        budgets = json.loads(PROMPT_BUDGET_PATH.read_text(encoding="utf-8")) if PROMPT_BUDGET_PATH.exists() else {}
        key = str(args.tool_stage)
        old = budgets.get(key, {})
        same = all(old.get(k) == result[k] for k in ("tool_catalog_version", "cases_sha256", "tokenizer_revision"))
        if args.engine == "default":
            # Keep the per-engine counts only if they were made against the same inputs.
            budgets[key] = {**result, **({"by_engine": old["by_engine"]} if same and "by_engine" in old else {})}
        else:
            if not same:
                raise SystemExit("count the default first (the per-engine count hangs off its entry)")
            budgets[key].setdefault("by_engine", {})[args.engine] = {
                k: result[k] for k in ("max_prompt_tokens", "max_case", "median_prompt_tokens")}
        PROMPT_BUDGET_PATH.write_text(json.dumps(budgets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {PROMPT_BUDGET_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
