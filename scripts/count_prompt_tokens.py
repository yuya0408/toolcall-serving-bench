#!/usr/bin/env python3
"""Count the longest prompt a cell will send, without a GPU (ADR-0005 2).

Builds every request exactly as scripts/run_cell.py does (build_tool_set,
trial_order, to_openai_tools_subset; latency queries carry the request-id
suffix) and renders it with Qwen3-8B's chat template at a pinned revision,
the way the servers do: apply_chat_template(..., tools=...,
add_generation_prompt=True, enable_thinking=False).

The 2026-10-18 smoke run checked this count against vLLM's prompt_tokens: equal
on all 90 latency requests (+-1 from the random request id); llama.cpp reported
4 tokens fewer on every request. The capacity check adds one allocation unit on
top to absorb such engine differences.

Needs `transformers` (tokenizer only; no torch) and the tokenizer files, which
it downloads from the Hub on first use:

    python scripts/count_prompt_tokens.py --tool-stage 10 --write-budget

--write-budget records the result in schema/prompt_budget.json, which the GPU
runs read (eval_bench/capacity.py). The entry carries the catalog version and a
digest of the cases, so editing either makes the runs refuse the stale count.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
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
# The latency pass's request ids are random hex, which moves the count by about
# +-1 token. Take the worst over a fixed set of ids, so the count is
# reproducible; the capacity check's one-unit margin covers the rest.
_TAG_DRAWS = 32
_TAG_SEED = 0


def _count(tokenizer, query: str, tool_set: list[dict]) -> int:
    # Render to text first, then tokenize, as the servers do (the template
    # already contains every special token, so none are added again).
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}], tools=tool_set,
        add_generation_prompt=True, enable_thinking=False, tokenize=False,
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def count(cases, tools, stage: int, tokenizer, trials: int) -> dict:
    """Per-case maximum over every request the cell can send for that case:
    the accuracy passes' `trials` shuffled orders and the latency pass's fixed
    seed-0 order with a request id appended."""
    per_case = {}
    for case in cases:
        tool_set = build_tool_set(case, tools, stage, seed=0)
        lengths = [_count(tokenizer, case["query"],
                          to_openai_tools_subset(tools, trial_order(case["id"], tool_set, stage, t)))
                   for t in range(trials)]
        fixed = to_openai_tools_subset(tools, tool_set)
        rng = random.Random(f"{_TAG_SEED}:{case['id']}")
        lengths += [_count(tokenizer, tagged_query(case["query"], f"{rng.getrandbits(32):08x}", 999_999), fixed)
                    for _ in range(_TAG_DRAWS)]
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
    p.add_argument("--write-budget", action="store_true",
                   help=f"record the result in {PROMPT_BUDGET_PATH.relative_to(_ROOT)}")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer  # imported late: only this script needs it
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision=args.revision)
    tools = load_tools(_ROOT / "schema" / _CATALOGS[args.tool_stage])
    cases = load_cases(args.data_dir)
    result = count(cases, tools, args.tool_stage, tokenizer, args.trials)
    result["tag_draws"] = _TAG_DRAWS
    result["tokenizer_revision"] = args.revision
    result["tool_catalog_version"] = json.loads(
        (_ROOT / "schema" / _CATALOGS[args.tool_stage]).read_text(encoding="utf-8"))["version"]
    result["cases_sha256"] = cases_digest(cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.write_budget:
        budgets = json.loads(PROMPT_BUDGET_PATH.read_text(encoding="utf-8")) if PROMPT_BUDGET_PATH.exists() else {}
        budgets[str(args.tool_stage)] = result
        PROMPT_BUDGET_PATH.write_text(json.dumps(budgets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {PROMPT_BUDGET_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
