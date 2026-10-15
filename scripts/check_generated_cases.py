"""Step 3 of the case-set expansion (docs/adr/0004-expand-case-set.md): find
the generated cases a human must look at.

    python scripts/check_generated_cases.py

Reads, from data_build/expand/:
  gold_calls.jsonl    step 1, the gold calls
  queries.jsonl       step 2, the query written from each gold call (and the negatives)
  blind_calls.jsonl   step 3, the call a second model chose from the query alone,
                      one line per id: {"id", "tool" (null = no call), "args"}
and writes review_queue.jsonl: every case, with why it needs review (if it
does) and whether it fell in the random sample.

A case is flagged when
  - the query writer or the blind model left a note on it;
  - the blind call disagrees with the gold call the way the scorer would
    count it: another tool, or a gold argument with another value after NFKC
    normalization (extra arguments are not penalized by the scorer, so they
    are not flagged either);
  - for a negative, the blind model called any tool;
  - its query is nearly identical to another one (character-bigram Jaccard).
Of the unflagged cases, SAMPLE_SIZE are drawn at random (fixed seed) for
human review, so the error rate among unreviewed cases can be bounded.
"""
from __future__ import annotations
import json
import pathlib
import random
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from eval_bench.scorers.separated import _normalize  # noqa: E402

BUILD = _ROOT / "data_build" / "expand"
SAMPLE_SIZE = 50
SAMPLE_SEED = 20261015
DUPLICATE_JACCARD = 0.8


def _jsonl(path: pathlib.Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def _bigrams(text: str) -> set[str]:
    t = _normalize(text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def disagreement(gold: dict | None, blind: dict) -> str | None:
    """Why the blind call does not match the gold call, or None if it does."""
    if gold is None:   # negative
        return None if blind["tool"] is None else f"negative, but the blind call used {blind['tool']}"
    if blind["tool"] != gold["expected_tool"]:
        return f"tool: gold {gold['expected_tool']}, blind {blind['tool']}"
    wrong = [k for k, v in gold["expected_args"]["exact"].items()
             if _normalize(blind.get("args", {}).get(k)) != _normalize(v)]
    if wrong:
        return "args: " + ", ".join(
            f"{k} gold {gold['expected_args']['exact'][k]!r} blind {blind.get('args', {}).get(k)!r}" for k in wrong)
    return None


def near_duplicates(queries: dict[str, dict]) -> dict[str, str]:
    ids = sorted(queries)
    grams = {i: _bigrams(queries[i]["query"]) for i in ids}
    out: dict[str, str] = {}
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            union = grams[a] | grams[b]
            if union and len(grams[a] & grams[b]) / len(union) >= DUPLICATE_JACCARD:
                out.setdefault(a, f"near-duplicate of {b}")
                out.setdefault(b, f"near-duplicate of {a}")
    return out


def main() -> int:
    gold = _jsonl(BUILD / "gold_calls.jsonl")
    queries = _jsonl(BUILD / "queries.jsonl")
    blind = _jsonl(BUILD / "blind_calls.jsonl")
    if set(queries) != set(blind):
        raise SystemExit(f"blind_calls.jsonl covers {len(blind)} ids, queries.jsonl {len(queries)}")

    dupes = near_duplicates(queries)
    rows = []
    for case_id in sorted(queries):
        reasons = [r for r in (disagreement(gold.get(case_id), blind[case_id]), dupes.get(case_id)) if r]
        if queries[case_id].get("writer_note"):
            reasons.append("writer note: " + queries[case_id]["writer_note"])
        if blind[case_id].get("note"):   # the blind model saw a second reading even if it picked the gold one
            reasons.append("blind note: " + blind[case_id]["note"])
        rows.append({"id": case_id, "query": queries[case_id]["query"], "flags": reasons, "sampled": False})

    unflagged = [r for r in rows if not r["flags"]]
    for r in random.Random(SAMPLE_SEED).sample(unflagged, min(SAMPLE_SIZE, len(unflagged))):
        r["sampled"] = True

    out = BUILD / "review_queue.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    flagged = sum(bool(r["flags"]) for r in rows)
    print(f"{len(rows)} cases: {flagged} flagged, {sum(r['sampled'] for r in rows)} sampled, "
          f"{len(rows) - flagged - sum(r['sampled'] for r in rows)} unreviewed -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
