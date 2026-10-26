"""Quantization non-inferiority of one engine (docs/adr/0008), offline from the
two raw-output logs of its greedy concurrency-1 cells.

    python scripts/noninferiority.py runs/<dir>/ni-vllm-bf16-c1/<run_id>.raw.jsonl \
        runs/<dir>/ni-vllm-awq-c1/<run_id>.raw.jsonl

Both arms share seed_base, so trial t presented the same tool order in both
and each (case, trial) is a matched pair. Per case, d = the 4bit arm's
selection rate minus the bf16 arm's (the mean over its trials); the estimate
is the mean of d over cases.

Verdict, fixed in ADR-0008 before the run:
  non-inferior   the one-sided 99.5% t lower bound on mean(d) > -2.6pt, and the
                 data are inside the range the simulation checked (p_d >= 4.5%,
                 rho <= 0.3), where that bound keeps the false 'non-inferior'
                 rate at or below 2.5%
  not shown      otherwise (including outside that range)
p_d is the share of matched pairs whose selection differs; rho is the ANOVA
estimate of the intra-case correlation of a pair being discordant.
arg_exact is reported per arm (pooled over applicable trials) and never
enters the verdict.
"""
from __future__ import annotations
import argparse
import json
import math
import pathlib
import statistics
import sys
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from count_flips import load_arm, _no_semantic_judge  # noqa: E402
from eval_bench.dataset.loader import load_cases  # noqa: E402
from eval_bench.scorers.separated import Prediction, score_case  # noqa: E402

MARGIN = 0.026         # docs/adr/0002, reused by docs/adr/0008
CONFIDENCE = 0.995     # one-sided; calibrated so the actual error rate is <= 2.5% (ADR-0008)
MIN_P_D = 0.045        # the range the calibration covered
MAX_RHO = 0.3


def t_quantile(q: float, df: int) -> float:
    """Student t quantile from the normal one (Abramowitz & Stegun 26.7.5).
    Stdlib only, like the rest of the harness; the series is accurate to well
    under 1e-4 at the degrees of freedom used here (199), and refused below 30."""
    if df < 30:
        raise ValueError(f"the expansion is not used below 30 degrees of freedom (got {df})")
    z = statistics.NormalDist().inv_cdf(q)
    return (z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3))


def intra_case_rho(discordant_per_case: list[int], trials: int) -> float:
    """ANOVA estimator of the intra-cluster correlation for binary outcomes
    with equal cluster sizes (here: was the pair discordant)."""
    k = len(discordant_per_case)
    p = sum(discordant_per_case) / (k * trials)
    msb = trials * sum((x / trials - p) ** 2 for x in discordant_per_case) / (k - 1)
    msw = sum(x * (1 - x / trials) for x in discordant_per_case) / (k * (trials - 1))
    denom = msb + (trials - 1) * msw
    return (msb - msw) / denom if denom > 0 else float("nan")


def noninferiority(base: dict[tuple[str, int], dict[str, Any]],
                   quant: dict[tuple[str, int], dict[str, Any]],
                   cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if base.keys() != quant.keys():
        raise ValueError("the two arms do not cover the same (case, trial) pairs")
    trials_per_case: dict[str, int] = {}
    for case_id, _ in base:
        trials_per_case[case_id] = trials_per_case.get(case_id, 0) + 1
    if len(set(trials_per_case.values())) != 1:
        raise ValueError("every case needs the same number of trials")
    trials = next(iter(trials_per_case.values()))

    d: dict[str, list[int]] = {}
    discordant: dict[str, int] = {}
    arg = {"base": [], "quant": []}
    for key in sorted(base):
        case = cases[key[0]]
        scores = {}
        for arm, rows in (("base", base), ("quant", quant)):
            got = rows[key]["decoded"]
            s = score_case(case, Prediction(got["tool"], got["args"]), _no_semantic_judge)
            scores[arm] = int(s.selection)
            if s.arg_exact is not None:
                arg[arm].append(int(s.arg_exact))
        d.setdefault(key[0], []).append(scores["quant"] - scores["base"])
        discordant[key[0]] = discordant.get(key[0], 0) + (scores["quant"] != scores["base"])

    per_case = [statistics.mean(v) for v in d.values()]
    n = len(per_case)
    mean = statistics.mean(per_case)
    se = statistics.stdev(per_case) / math.sqrt(n)
    lower = mean - t_quantile(CONFIDENCE, n - 1) * se
    flips = list(discordant.values())
    p_d = sum(flips) / (n * trials)
    rho = intra_case_rho(flips, trials) if p_d else float("nan")
    in_range = p_d >= MIN_P_D and not math.isnan(rho) and rho <= MAX_RHO
    return {
        "n_cases": n, "trials": trials,
        "selection_diff": mean, "lower_bound": lower, "confidence": CONFIDENCE, "margin": -MARGIN,
        "p_d": p_d, "rho": rho, "cases_with_a_flip": sum(1 for x in flips if x),
        "in_checked_range": in_range,
        "verdict": "non-inferior" if in_range and lower > -MARGIN else "not shown",
        "arg_exact_mean": {arm: (statistics.mean(v) if v else None) for arm, v in arg.items()},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base", type=pathlib.Path, help="raw log of the bf16 arm")
    p.add_argument("quant", type=pathlib.Path, help="raw log of the 4bit arm")
    p.add_argument("--data-dir", type=pathlib.Path, default=_ROOT / "data")
    args = p.parse_args(argv)

    cases = {c["id"]: c for c in load_cases(args.data_dir)}
    result = noninferiority(load_arm(args.base), load_arm(args.quant), cases)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
