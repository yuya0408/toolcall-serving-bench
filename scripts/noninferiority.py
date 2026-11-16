"""Quantization non-inferiority of one engine (docs/adr/0008), offline from the
two raw-output logs of its greedy concurrency-1 cells.

    python scripts/noninferiority.py runs/<dir>/ni-vllm-bf16-c1/<run_id>.raw.jsonl \
        runs/<dir>/ni-vllm-awq-c1/<run_id>.raw.jsonl

With --batching runs/<dir> [runs/<rerun dir> ...], the phase-2 accuracy gate instead
(docs/adr/0009 5, option F). The verdicts are in eval_bench/noninferiority.py.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from eval_bench.dataset.loader import load_cases  # noqa: E402
from eval_bench.flips import load_arm  # noqa: E402
from eval_bench.noninferiority import batching_gate, load_batching_arms, noninferiority  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base", type=pathlib.Path, nargs="?", help="raw log of the bf16 arm")
    p.add_argument("quant", type=pathlib.Path, nargs="?", help="raw log of the 4bit arm")
    p.add_argument("--batching", type=pathlib.Path, nargs="+", metavar="RUN_DIR",
                   help="the phase-2 gate over RUN_DIR/ni-<engine>-<quant>-c1 (docs/adr/0009); "
                        "the main run's folder first, then any rerun's")
    p.add_argument("--exclude", nargs="*", default=[], metavar="ENGINE-QUANT",
                   help="with --batching: configs whose ni cell failed again after a rerun (docs/adr/0009 5)")
    p.add_argument("--data-dir", type=pathlib.Path, default=_ROOT / "data")
    args = p.parse_args(argv)
    if (args.batching is None) == (args.base is None or args.quant is None):
        p.error("give either base and quant, or --batching")

    cases = {c["id"]: c for c in load_cases(args.data_dir)}
    if args.batching is not None:
        result = batching_gate(load_batching_arms(args.batching, cases, tuple(args.exclude)))
        result["excluded"] = list(args.exclude)
    else:
        result = noninferiority(load_arm(args.base), load_arm(args.quant), cases)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
