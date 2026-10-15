"""Step 4 of the case-set expansion (docs/adr/0004-expand-case-set.md): turn
the reviewed build files into evaluation cases.

    python scripts/build_generated_cases.py    # writes data/generated/generated.jsonl

Reads, from data_build/expand/: gold_calls.jsonl, queries.jsonl,
review_queue.jsonl (scripts/check_generated_cases.py), review.jsonl, the
human decisions, one line per reviewed id:

    {"id": "case_1042", "decision": "keep" | "fix" | "drop",
     "query": "<corrected query, decision fix only>",
     "gold_add": {"<arg>": "<value>"},   # optional, decision fix only
     "note": "<why>"}

and blind_recheck.jsonl, the blind check (same shape as blind_calls.jsonl)
rerun on the corrected queries only.

Every flagged or sampled case must have a decision. A fix rewrites the query
and may add gold arguments that the corrected query states, but never
changes or removes a value the seeded script chose; anything else wrong with
a gold call is a drop. Every fixed case must pass the blind check again
against its corrected gold call. Each case records how far it was checked
(generation.review) and what a human changed (generation.human_edit), so
metrics can be restricted to human-reviewed or unedited cases.
"""
from __future__ import annotations
import json
import pathlib
import random
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from eval_bench.dataset.tools import load_tools, validate_case_against_tools  # noqa: E402

BUILD = _ROOT / "data_build" / "expand"
OUT = _ROOT / "data" / "generated" / "generated.jsonl"
QUERY_MODEL = "claude-sonnet-5-5"
CHECK_MODEL = "claude-opus-5-5"
NEGATIVE_DISTRACTOR_SEED = 20261015


def _jsonl(path: pathlib.Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def main() -> int:
    sys.path.insert(0, str(_ROOT / "scripts"))
    from generate_gold_calls import SEED  # the gold calls' seed, recorded on every case
    from check_generated_cases import disagreement

    tools = load_tools(_ROOT / "schema" / "tools.json")
    gold = _jsonl(BUILD / "gold_calls.jsonl")
    queries = _jsonl(BUILD / "queries.jsonl")
    queue = _jsonl(BUILD / "review_queue.jsonl")
    review = _jsonl(BUILD / "review.jsonl")
    recheck = _jsonl(BUILD / "blind_recheck.jsonl")

    needs_review = [i for i, r in queue.items() if r["flags"] or r["sampled"]]
    missing = [i for i in needs_review if i not in review]
    if missing:
        raise SystemExit(f"{len(missing)} flagged/sampled cases have no decision in review.jsonl, e.g. {missing[:5]}")

    rng = random.Random(NEGATIVE_DISTRACTOR_SEED)
    names = sorted(tools)
    cases = []
    for case_id in sorted(queries):
        decision = review.get(case_id, {"decision": "keep"})
        if decision["decision"] == "drop":
            continue
        fixed = decision["decision"] == "fix"
        query = decision["query"] if fixed else queries[case_id]["query"]
        g = gold.get(case_id)
        added = decision.get("gold_add", {}) if fixed else {}
        if added:
            if g is None or set(added) & set(g["expected_args"]["exact"]):
                raise SystemExit(f"{case_id}: gold_add may only add arguments to a gold call, not change them")
            g = {**g, "expected_args": {**g["expected_args"],
                                        "exact": {**g["expected_args"]["exact"], **added}}}
        if fixed:
            if case_id not in recheck:
                raise SystemExit(f"{case_id}: fixed but not in blind_recheck.jsonl")
            why = disagreement(g, recheck[case_id])
            if why:
                raise SystemExit(f"{case_id}: the blind recheck disagrees with the corrected case ({why})")
        if queue[case_id]["flags"]:
            status = "flagged_reviewed"
        elif queue[case_id]["sampled"]:
            status = "sampled_reviewed"
        else:
            status = "unreviewed"
        case = {
            "id": case_id,
            "query": query,
            "expected_tool": g["expected_tool"] if g else None,
            "expected_args": g["expected_args"] if g else {"exact": {}, "semantic": {}},
            # Negatives have no neighbours by construction; 3 random tools keep the field's shape.
            "distractor_tools": g["distractor_tools"] if g else rng.sample(names, 3),
            "negative": g is None,
            "provenance": "generated",
            "generation": {"gold_seed": SEED, "query_model": QUERY_MODEL,
                           "check_model": CHECK_MODEL, "review": status,
                           "human_edit": "query_and_gold" if added else "query" if fixed else "none"},
        }
        validate_case_against_tools(case, tools)
        cases.append(case)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {OUT} ({len(queries) - len(cases)} dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
