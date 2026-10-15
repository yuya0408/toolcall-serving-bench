"""How many cases the third pass needs, and which interval keeps its promise
(docs/adr/0002-equivalence-margin.md).

    python scripts/power_simulation.py                       # the planning table
    python scripts/power_simulation.py --pd 0.08 --rho 0.4   # one pilot's estimates

The third pass compares concurrency 1 and 16 on the same cases (greedy, T
shuffled trials per case) and calls them equivalent when a 95% interval for
the selection-rate difference lies inside +-2.6pt -- two one-sided tests at
2.5% each (Schuirmann's TOST). In the statistics literature this data shape
is "clustered matched-pair binary data": a case is a cluster, its T trials
are the matched pairs.

Simulation design (Morris, White & Crowther 2019, ADEMP):
  Aims        Type I error at the margin and power at no difference, per method.
  Data        Each (case, trial) pair is discordant between the arms with
              probability p_i; p_i ~ Beta with mean p_d and intra-case
              correlation rho (rho=0: every case shares p_d). A discordant
              pair favors the loaded arm with probability 1/2 + delta/(2 p_d),
              so the true difference is delta.
  Estimand    delta, the difference in pooled selection rate (loaded - serial).
  Methods     With equal trials per case, the pooled difference is the mean of
              the per-case differences d_k, so every method works on d_k:
                bootstrap  repetition._paired_cluster_bootstrap_diff_ci (current):
                           percentile CI from resampling cases
                t          d_mean +- t(K-1) * sd(d)/sqrt(K): Schuirmann's TOST on
                           cases. With equal cluster sizes this is also
                           Obuchowski's (1998) cluster-adjusted variance, centered
                           at the estimate
                shifted    each one-sided test uses Obuchowski's variance centered
                           at its null value (-margin / +margin) instead of 0
                           [own generalization of the H0: delta=0 form; not checked
                           against a published non-inferiority version]
                yang       each one-sided test averages the estimate-centered and
                           null-centered variances, as Yang, Sun & Hardin (2010) do
                           for H0: delta=0 [same caveat]
  Performance size  = P(equivalent | delta = +-margin), must be <= 0.025 per side
              power = P(equivalent | delta = 0)
              Each is reported with its Monte Carlo standard error.

The bootstrap is exact but fast: resampling cases uniformly is a multinomial
draw over the (at most 2T+1) distinct d_k values, weighted by their counts.
"""
from __future__ import annotations
import argparse
import math
import sys

import numpy as np
from scipy import stats

MARGIN = 0.026          # ADR-0002
TRIALS = 5              # scripts/run_cell.py --n default
N_RESAMPLES = 2000      # repetition.py default
ALPHA = 0.05            # 95% two-sided interval = 2.5% per one-sided test
METHODS = ("bootstrap", "t", "shifted", "yang")


def simulate_case_diffs(rng: np.random.Generator, n_cases: int, p_d: float, rho: float,
                        delta: float, trials: int = TRIALS) -> np.ndarray:
    """Per-case (loaded - serial) differences in selection rate."""
    if rho > 0:
        a = p_d * (1 - rho) / rho
        b = (1 - p_d) * (1 - rho) / rho
        p = rng.beta(a, b, size=n_cases)
    else:
        p = np.full(n_cases, p_d)
    q = 0.5 + delta / (2 * p_d)
    if not 0 <= q <= 1:
        raise ValueError(f"delta={delta} is impossible with p_d={p_d} (|delta| must be <= p_d)")
    discordant = rng.binomial(trials, p)
    better = rng.binomial(discordant, q)
    return (2 * better - discordant) / trials


def bootstrap_ci(rng: np.random.Generator, case_diffs: np.ndarray,
                 n_resamples: int = N_RESAMPLES, alpha: float = ALPHA) -> tuple[float, float]:
    """Same percentile CI as repetition._paired_cluster_bootstrap_diff_ci."""
    values, counts = np.unique(case_diffs, return_counts=True)
    n = len(case_diffs)
    draws = rng.multinomial(n, counts / n, size=n_resamples)
    means = np.sort(draws @ values / n)
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)
    return float(means[lo_idx]), float(means[hi_idx])


def _one_sided_pair(d: np.ndarray, var_lower: float, var_upper: float, alpha: float) -> bool:
    """Both one-sided z tests reject: H0 delta <= -margin and H0 delta >= +margin."""
    z = stats.norm.ppf(1 - alpha / 2)
    mean = d.mean()
    if var_lower <= 0 or var_upper <= 0:
        return False
    return (mean + MARGIN) / math.sqrt(var_lower) > z and (MARGIN - mean) / math.sqrt(var_upper) > z


def is_equivalent(method: str, d: np.ndarray, rng: np.random.Generator, alpha: float = ALPHA) -> bool:
    k = len(d)
    mean = d.mean()
    if method == "bootstrap":
        lo, hi = bootstrap_ci(rng, d, alpha=alpha)
        return lo >= -MARGIN and hi <= MARGIN          # equivalence_verdict's "equivalent"
    if method == "t":
        se = d.std(ddof=1) / math.sqrt(k)
        half = stats.t.ppf(1 - alpha / 2, k - 1) * se
        return mean - half >= -MARGIN and mean + half <= MARGIN
    centered = ((d - mean) ** 2).sum()
    at_lower = ((d + MARGIN) ** 2).sum()                # variance under H0: delta = -margin
    at_upper = ((d - MARGIN) ** 2).sum()                # ... under H0: delta = +margin
    scale = k * (k - 1)
    if method == "shifted":
        return _one_sided_pair(d, at_lower / scale, at_upper / scale, alpha)
    if method == "yang":
        return _one_sided_pair(d, (centered + at_lower) / 2 / scale,
                               (centered + at_upper) / 2 / scale, alpha)
    raise ValueError(method)


def rates(n_cases: int, p_d: float, rho: float, delta: float, n_sims: int,
          seed: int = 0, methods=METHODS) -> dict[str, float]:
    """P(equivalent) per method, every method on the same simulated datasets."""
    rng = np.random.default_rng(seed)
    hits = dict.fromkeys(methods, 0)
    for _ in range(n_sims):
        d = simulate_case_diffs(rng, n_cases, p_d, rho, delta)
        for m in methods:
            hits[m] += is_equivalent(m, d, rng)
    return {m: h / n_sims for m, h in hits.items()}


def mcse(p: float, n_sims: int) -> float:
    return math.sqrt(p * (1 - p) / n_sims)


_GRID = [14, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 700, 1000]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pd", type=float, nargs="+", default=[0.03, 0.05, 0.10, 0.20])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    args = ap.parse_args(argv)

    print(f"margin +-{MARGIN}, {TRIALS} trials/case, {args.sims} sims per point; "
          f"MCSE of a rate near 0.025 = {mcse(0.025, args.sims):.4f}")
    print("p_d   rho  method     | n for power 80% | 90% | size at -margin / +margin, at the 80% n")
    for p_d in args.pd:
        for rho in args.rho:
            n80 = dict.fromkeys(args.methods)
            n90 = dict.fromkeys(args.methods)
            for n in _GRID:
                pending = [m for m in args.methods if n90[m] is None]
                if not pending:
                    break
                power = rates(n, p_d, rho, 0.0, args.sims, methods=pending)
                for m in pending:
                    if n80[m] is None and power[m] >= 0.80:
                        n80[m] = n
                    if power[m] >= 0.90:
                        n90[m] = n
            for m in args.methods:
                size = "n/a"
                if n80[m] is not None and p_d >= MARGIN:
                    lo = rates(n80[m], p_d, rho, -MARGIN, args.sims, seed=1, methods=[m])[m]
                    hi = rates(n80[m], p_d, rho, +MARGIN, args.sims, seed=2, methods=[m])[m]
                    size = f"{lo:.3f} / {hi:.3f}"
                fmt = lambda n: f">{_GRID[-1]}" if n is None else str(n)  # noqa: E731
                print(f"{p_d:.2f}  {rho:.1f}  {m:<10} | {fmt(n80[m]):>15} | {fmt(n90[m]):>4} | {size}")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
