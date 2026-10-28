"""Quantization non-inferiority of one engine (docs/adr/0008), offline from the
two raw-output logs of its greedy concurrency-1 cells.

    python scripts/noninferiority.py runs/<dir>/ni-vllm-bf16-c1/<run_id>.raw.jsonl \
        runs/<dir>/ni-vllm-awq-c1/<run_id>.raw.jsonl

With --batching runs/<dir> [runs/<rerun dir> ...], the phase-2 accuracy gate instead (docs/adr/0009 5,
option F; see batching_gate). Phase 1 stays under the ADR-0008 verdict below.

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
POOLED_CONFIDENCE = 0.999   # docs/adr/0009 5; worst simulated error 2.3% (scripts/gate_simulation.py)
BATCHING_ENGINES = ("vllm", "sglang", "trt")
BATCHING_QUANT = ("awq", "fp8")


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


def _selection(arm: dict[tuple[str, int], dict[str, Any]],
               cases: dict[str, dict[str, Any]]) -> dict[tuple[str, int], int]:
    return {key: int(score_case(cases[key[0]], Prediction(row["decoded"]["tool"], row["decoded"]["args"]),
                                _no_semantic_judge).selection) for key, row in arm.items()}


def pooled_noninferiority(refs: list[dict[tuple[str, int], int]],
                          test: dict[tuple[str, int], int]) -> dict[str, Any]:
    """docs/adr/0009 5 (option F): the test arm against the per-(case, trial) mean
    of one or more reference arms, all scored (1 = right tool). Per case, d = the
    mean over trials of (test - mean of the references); claimed when the one-sided
    99.9% t lower bound on mean(d) > -2.6pt. No range rule: the level was set so the
    simulated error stays <= 2.5% for p_d 0-12% and rho <= 0.3 (rho is a limit, not
    checked). p_d (vs each reference) and rho are reported, not used."""
    if not refs or any(r.keys() != test.keys() for r in refs):
        raise ValueError("every arm must cover the same (case, trial) pairs")
    d: dict[str, list[float]] = {}
    for key in sorted(test):
        d.setdefault(key[0], []).append(test[key] - statistics.mean(r[key] for r in refs))
    if len({len(v) for v in d.values()}) != 1:
        raise ValueError("every case needs the same number of trials")
    trials = len(next(iter(d.values())))
    per_case = [statistics.mean(v) for v in d.values()]
    n = len(per_case)
    mean = statistics.mean(per_case)
    sd = statistics.stdev(per_case)
    lower = mean - t_quantile(POOLED_CONFIDENCE, n - 1) * sd / math.sqrt(n) if sd else mean
    p_d, rho = [], []
    for r in refs:
        flips: dict[str, int] = {}
        for key in test:
            flips[key[0]] = flips.get(key[0], 0) + (test[key] != r[key])
        share = sum(flips.values()) / (n * trials)
        p_d.append(share)
        rho.append(intra_case_rho(list(flips.values()), trials) if share else float("nan"))
    return {"n_cases": n, "trials": trials, "n_references": len(refs),
            "selection_diff": mean, "lower_bound": lower, "confidence": POOLED_CONFIDENCE, "margin": -MARGIN,
            "p_d_vs_each_reference": p_d, "rho_vs_each_reference": rho,
            "verdict": "non-inferior" if lower > -MARGIN else "not shown"}


def batching_gate(arms: dict[str, dict[tuple[str, int], int]]) -> dict[str, Any]:
    """The phase-2 gate (docs/adr/0009 5), on scored arms named '<engine>-<quant>'
    (missing = not measured, e.g. dropped at the smoke apply).

    1. Each bf16 against the other bf16s (fixed: not re-pooled by their verdicts).
       A lone bf16 is the reference and passes untested.
    2. Each quantized config against the bf16s that passed, whether or not its own
       bf16 did; its difference from its own bf16 is reported, not judged.
    3. No bf16 passed: nothing passes (the bf16s disagree); nothing is selected."""
    bf16 = [f"{e}-bf16" for e in BATCHING_ENGINES if f"{e}-bf16" in arms]
    out: dict[str, Any] = {"bf16": {}, "quantized": {}}
    for name in bf16:
        others = [o for o in bf16 if o != name]
        if not others:
            out["bf16"][name] = {"references": [], "verdict": "reference"}
            continue
        out["bf16"][name] = {"references": others,
                             **pooled_noninferiority([arms[o] for o in others], arms[name])}
    passed = [n for n, r in out["bf16"].items() if r["verdict"] in ("non-inferior", "reference")]
    out["passed_bf16"] = passed
    for e in BATCHING_ENGINES:
        for q in BATCHING_QUANT:
            name = f"{e}-{q}"
            if name not in arms:
                continue
            if not passed:
                out["quantized"][name] = {"references": [], "verdict": "not shown (no bf16 passed)"}
                continue
            r = {"references": passed, **pooled_noninferiority([arms[p] for p in passed], arms[name])}
            own = f"{e}-bf16"
            if own in arms:   # descriptive only
                r["diff_vs_own_bf16"] = statistics.mean(arms[name][k] - arms[own][k] for k in arms[name])
            out["quantized"][name] = r
    out["passing"] = passed + [n for n, r in out["quantized"].items() if r["verdict"] == "non-inferior"]
    return out


def load_batching_arms(run_dirs: list[pathlib.Path], cases: dict[str, dict[str, Any]],
                       exclude: tuple[str, ...] = ()) -> dict[str, dict[tuple[str, int], int]]:
    """The scored greedy arms ni-<engine>-<quant>-c1, one raw log each, from the
    main run's folder and then any rerun's (a later folder's run of a cell wins).

    An arm the run's status.jsonl does not list was not in the cell list (an
    engine dropped at the smoke apply) and is left out. An arm it lists must have
    finished ok: a failed one is rerun on a new apply before the gate is read
    (docs/adr/0009 5), and only a config that fails again is left out, by name
    (exclude), so no arm goes missing without saying so."""
    outcome, where = {}, {}
    for run_dir in run_dirs:
        status_path = run_dir / "status.jsonl"
        if not status_path.exists():
            raise ValueError(f"{status_path} not found: the gate reads which cells finished from it")
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                outcome[row["name"]] = row["outcome"]   # the last run of a cell counts
                where[row["name"]] = run_dir
    arms = {}
    for e in BATCHING_ENGINES:
        for q in ("bf16",) + BATCHING_QUANT:
            name = f"ni-{e}-{q}-c1"
            if f"{e}-{q}" in exclude or name not in outcome:
                continue
            if outcome[name] != "ok":
                raise ValueError(f"{name} finished as {outcome[name]}: rerun it on a new apply (docs/adr/0009 5), "
                                 f"or pass --exclude {e}-{q} if it failed again")
            logs = sorted((where[name] / name).glob("*.raw.jsonl"))
            if len(logs) != 1:
                raise ValueError(f"{where[name] / name}: {len(logs)} raw logs, expected one")
            arms[f"{e}-{q}"] = _selection(load_arm(logs[0]), cases)
    return arms


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
