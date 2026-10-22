"""Count the cases whose correctness flipped between the two arms of a third
pass (docs/adr/0002), offline from the two raw-output logs.

    python scripts/count_flips.py runs/<dir>/third-vllm-awq-c1/<run_id>.raw.jsonl \
        runs/<dir>/third-vllm-awq-c16/<run_id>.raw.jsonl

A flip is a (case, trial) whose selection is correct in one arm and wrong in
the other. Both arms share seed_base, so trial t presented the same tool
order in both (repetition.trial_order) and the pairing is exact. ADR-0002's
branch after the pilot reads the number of cases with at least one flipped
trial: 0 -> option A is a candidate, several -> option C. That count is what
this script prints first; option A's one-sided 97.5% Clopper-Pearson upper
bound on the share of such cases is printed next to it for reference.

Two weaker kinds of change are counted separately, since they show the batch
changed the computation even where the score did not move: cases whose
decoded (tool, args) changed without a flip, and cases whose raw output
changed at all (e.g. the wording of a declining answer), ignoring the
per-request tool-call ids and the usage block.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from eval_bench.dataset.loader import load_cases  # noqa: E402
from eval_bench.latency import clopper_pearson_one_sided  # noqa: E402
from eval_bench.scorers.separated import Prediction, score_case  # noqa: E402

# ADR-0002 option A: one-sided 97.5% upper bound, equivalent if <= 2.6%.
_OPTION_A_CONFIDENCE = 0.975
_MARGIN = 0.026


def _no_semantic_judge(expected: str, predicted: str) -> bool:
    raise RuntimeError("a semantic-scored field was scored; this catalog has none")


def comparable_output(raw: str) -> Any:
    """The raw output minus what differs between any two requests: the usage
    block and the ids the server gives each tool call."""
    try:
        out = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw
    if not isinstance(out, dict):
        return out
    out.pop("_usage", None)
    for call in out.get("tool_calls") or []:
        if isinstance(call, dict):
            call.pop("id", None)
    return out


def load_arm(path: pathlib.Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["trial"] is None:
            raise ValueError(f"{path}: {row['case_id']} has a row whose trial could not be recovered")
        key = (row["case_id"], row["trial"])
        if key in rows:
            raise ValueError(f"{path}: duplicate row for {key}")
        rows[key] = {"decoded": row["decoded"], "output": comparable_output(row["raw_output"])}
    return rows


def count_flips(base: dict[tuple[str, int], dict[str, Any]],
                loaded: dict[tuple[str, int], dict[str, Any]],
                cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if base.keys() != loaded.keys():
        raise ValueError("the two arms do not cover the same (case, trial) pairs")
    flipped_pairs: list[dict[str, Any]] = []
    changed_cases: set[str] = set()
    raw_changed_cases: set[str] = set()
    for case_id, trial in sorted(base):
        if base[(case_id, trial)]["output"] != loaded[(case_id, trial)]["output"]:
            raw_changed_cases.add(case_id)
        a, b = base[(case_id, trial)]["decoded"], loaded[(case_id, trial)]["decoded"]
        if a != b:
            changed_cases.add(case_id)
        case = cases[case_id]
        sa = score_case(case, Prediction(a["tool"], a["args"]), _no_semantic_judge).selection
        sb = score_case(case, Prediction(b["tool"], b["args"]), _no_semantic_judge).selection
        if sa != sb:
            flipped_pairs.append({"case_id": case_id, "trial": trial,
                                  "direction": "lost_under_load" if sa else "gained_under_load",
                                  "base": a, "loaded": b})
    flipped_cases = sorted({p["case_id"] for p in flipped_pairs})
    n_cases = len({c for c, _ in base})
    _, upper = clopper_pearson_one_sided(len(flipped_cases), n_cases, _OPTION_A_CONFIDENCE)
    return {
        "n_cases": n_cases,
        "n_pairs": len(base),
        "flipped_cases": flipped_cases,
        "flipped_pairs": flipped_pairs,
        "lost_under_load": sum(p["direction"] == "lost_under_load" for p in flipped_pairs),
        "gained_under_load": sum(p["direction"] == "gained_under_load" for p in flipped_pairs),
        "changed_without_flip": sorted(changed_cases - set(flipped_cases)),
        "raw_changed_cases": sorted(raw_changed_cases),
        "option_a_upper_975": upper,
        "option_a_equivalent": upper <= _MARGIN,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base", type=pathlib.Path, help="raw log of the concurrency-1 arm")
    p.add_argument("loaded", type=pathlib.Path, help="raw log of the loaded arm")
    p.add_argument("--data-dir", type=pathlib.Path, default=_ROOT / "data")
    args = p.parse_args(argv)

    cases = {c["id"]: c for c in load_cases(args.data_dir)}
    result = count_flips(load_arm(args.base), load_arm(args.loaded), cases)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
