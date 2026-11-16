"""Count the cases whose correctness flipped between the two arms of a third
pass (docs/adr/0002), offline from the two raw-output logs.

    python scripts/count_flips.py runs/<dir>/third-vllm-awq-c1/<run_id>.raw.jsonl \
        runs/<dir>/third-vllm-awq-c16/<run_id>.raw.jsonl

The number of cases with at least one flipped trial is printed first; the
counting is in eval_bench/flips.py.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from eval_bench.dataset.loader import load_cases  # noqa: E402
from eval_bench.flips import count_flips, load_arm  # noqa: E402


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
