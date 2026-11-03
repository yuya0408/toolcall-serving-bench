"""Write the CI gate's golden files (docs/adr/0010-ci-golden-gate.md).

    python scripts/update_golden.py --from-runs runs/     # first time: copy the answers in
    python scripts/update_golden.py                       # after an intended G2/G4 change

--from-runs reads the five cells in eval_bench/golden.py's CELLS from a runs/
folder (the main runs, or Release runs-2026-10 unpacked) and writes
tests/golden/<engine>.answers.jsonl.gz and cells.json. It refuses unless the
current code reproduces the records (G1 tool order, G3 decoding) and the
rescored selection mean equals the manifest's.

Both forms then rewrite tests/golden/<engine>.expected.jsonl.gz from the
current code and print every (case, trial) whose request or score changed.
The answers are run records and are never regenerated after the first copy.
"""
from __future__ import annotations
import argparse
import glob
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from eval_bench import golden  # noqa: E402


def _copy_answers(runs: pathlib.Path, cases: dict, tools: dict) -> None:
    cells = {}
    for engine, rel in golden.CELLS.items():
        cell_dir = runs / rel
        raws = glob.glob(str(cell_dir / "*.raw.jsonl"))
        manifests = [m for m in glob.glob(str(cell_dir / "*Z.json"))]
        if len(raws) != 1 or len(manifests) != 1:
            sys.exit(f"{cell_dir}: expected one raw log and one manifest")
        manifest = json.loads(pathlib.Path(manifests[0]).read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in pathlib.Path(raws[0]).read_text(encoding="utf-8").splitlines()]
        answers = sorted(({k: r[k] for k in golden.ANSWER_FIELDS} for r in rows),
                         key=lambda a: (a["case_id"], a["trial"]))
        cell = {"source": rel, "run_id": manifest["run_id"], "engine": engine,
                "model": manifest["cell"]["model"], "sampling_params": manifest["cell"]["sampling_params"],
                "tool_stage": manifest["cell"]["tool_stage"], "seed_base": manifest["repetition"]["seed_base"],
                "selection_mean": manifest["results"]["selection_mean"]}
        bad = golden.check_g1_tool_order(cell, answers, cases, tools) + golden.check_g3_decoding(engine, answers)
        if bad:
            sys.exit(f"{engine}: the current code does not reproduce the run record ({len(bad)} rows), e.g. {bad[0]}")
        mean = golden.selection_mean(golden.expected_rows(engine, cell, answers, cases, tools))
        if round(mean, 3) != cell["selection_mean"]:
            sys.exit(f"{engine}: rescored selection {mean:.4f} != manifest {cell['selection_mean']}")
        golden.write_jsonl_gz(golden.GOLDEN_DIR / f"{engine}.answers.jsonl.gz", answers)
        cells[engine] = cell
        print(f"{engine}: {len(answers)} answers from {rel}, selection {mean:.3f} matches the manifest")
    (golden.GOLDEN_DIR / "cells.json").write_bytes(
        (json.dumps(cells, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8"))


def _write_expected(cases: dict, tools: dict) -> int:
    changed = 0
    cells = json.loads((golden.GOLDEN_DIR / "cells.json").read_text(encoding="utf-8"))
    for engine, cell in cells.items():
        answers = golden.read_jsonl_gz(golden.GOLDEN_DIR / f"{engine}.answers.jsonl.gz")
        new = golden.expected_rows(engine, cell, answers, cases, tools)
        path = golden.GOLDEN_DIR / f"{engine}.expected.jsonl.gz"
        old = golden.read_jsonl_gz(path) if path.exists() else []
        diffs = [(n["case_id"], n["trial"], field) for o, n in zip(old, new)
                 for field in ("request", "score") if o[field] != n[field]]
        for case_id, trial, field in diffs:
            print(f"  {engine} {case_id} trial {trial}: {field} changed")
        changed += len(diffs)
        golden.write_jsonl_gz(path, new)
        print(f"{engine}: {len(new)} expected rows, {len(diffs)} changed"
              + ("" if old else " (new file)"))
    return changed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from-runs", type=pathlib.Path, metavar="RUNS_DIR",
                   help="copy the answers in from the main runs (first time only)")
    args = p.parse_args(argv)
    cases, tools = golden.harness_inputs()
    golden.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    if args.from_runs is not None:
        _copy_answers(args.from_runs, cases, tools)
    _write_expected(cases, tools)
    return 0


if __name__ == "__main__":
    sys.exit(main())
