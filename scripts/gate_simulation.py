"""Calibrate the one confidence level of the phase-2 accuracy gate
(docs/adr/0009 5, option F; scripts/noninferiority.py batching_gate).

    python scripts/gate_simulation.py 4000            # both tables, ~30 min
    python scripts/gate_simulation.py 4000 --table spread

The gate compares a test arm (one config, greedy, 200 cases x 5 trials) with the
per-(case, trial) mean of 1-3 reference arms (bf16 of other engines) and claims
non-inferiority when the one-sided t lower bound on the mean per-case difference
is above -2.6pt. Unlike ADR-0008 there is no range rule, so one level has to hold
over every disagreement rate that can occur.

Simulation design (Morris, White & Crowther 2019, ADEMP):
  Aims        Error (claim at a true difference of -2.6pt) and power (claim at 0),
              per level, for 1-3 references.
  Data        Each case has a reference answer Z_c ~ Bern(0.75) and an instability
              h_c ~ Beta(mean mu, intra-case correlation rho). A reference arm flips
              Z_c at each trial with probability h_c, independently per arm, so two
              references disagree on about 2 mu of the pairs (pd_ref). The test arm
              flips Z_c = 0 -> 1 with probability min(1, k0 h_c) and 1 -> 0 with
              min(1, k1 h_c); k0 = k sets how unstable it is, and k1 is solved so the
              expected difference is exactly -2.6pt (error) or 0 (power).
  Estimand    The difference in selection rate, test minus the references' mean.
  Methods     One-sided t on the per-case differences, levels 99.5-99.95%.
  Performance P(claim), with Monte Carlo SE sqrt(p(1-p)/sims) (0.25pt near 2.5%).
Tables:
  spread  pd_ref 1-8%, the test arm k = 1 (as unstable as a reference) or 2
  tight   the references agree closely (pd_ref 0.2-2%) and the test arm
          disagrees with one of them at pd_test 3-8% (a quantized config)
Error cells are left out where the test arm cannot be 2.6pt worse
(pd_test < 2.6%, or no k1 reaches it).
Result (2026-10-27, 4000 sims): worst error at 99.9% = 2.3%, so 99.9% is used.
The verdicts equal pooled_noninferiority's on every one of the 181 x 4000 error
datasets (2026-11-13, docs/adr/0009 A3; tests/test_gate_simulation.py keeps checking).
"""
import argparse
import math
import sys

import numpy as np
from scipy import stats

M = 0.026
CASES, TRIALS = 200, 5
ACC = 0.75
LEVELS = (0.995, 0.997, 0.998, 0.999, 0.9995)


def beta(rng, mu, rho, size):
    if rho == 0:
        return np.full(size, mu)
    return rng.beta(mu * (1 - rho) / rho, (1 - mu) * (1 - rho) / rho, size=size)


def solve_k1(mu, rho, k0, target=-M):
    """k1 so that E[test] - E[ref] = target, clipping flip probabilities at 1."""
    h = beta(np.random.default_rng(99), mu, rho, 400_000)
    ref = ACC * (1 - h.mean()) + (1 - ACC) * h.mean()

    def diff(k1):
        return ACC * (1 - np.minimum(1, k1 * h).mean()) + (1 - ACC) * np.minimum(1, k0 * h).mean() - ref

    lo, hi = 0.0, 1.0
    while diff(hi) > target:
        hi *= 2
        if hi > 1e6:
            return None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if diff(mid) > target else (lo, mid)
    return hi


def arms(rng, mu, rho, k0, k1, refs, sims, chunk=500):
    """The simulated arms, `chunk` datasets at a time: (test, [reference arms]),
    each a bool array (datasets, CASES, TRIALS) with True = right tool."""
    for start in range(0, sims, chunk):
        n = min(chunk, sims - start)
        z = rng.random((n, CASES, 1)) < ACC
        h = beta(rng, mu, rho, (n, CASES, 1))
        ref_arms = [z ^ (rng.random((n, CASES, TRIALS)) < h) for _ in range(refs)]
        p = np.where(z, np.minimum(1, k1 * h), np.minimum(1, k0 * h))
        yield z ^ (rng.random((n, CASES, TRIALS)) < p), ref_arms


def tstats(rng, mu, rho, k0, k1, refs, sims, chunk=500):
    """The one-sided t statistic of noninferiority.pooled_noninferiority, per
    dataset; claimed when above the t quantile (tests/test_gate_simulation.py
    checks that the verdicts agree)."""
    out = []
    for test, ref_arms in arms(rng, mu, rho, k0, k1, refs, sims, chunk):
        ref = np.zeros(test.shape)
        for a in ref_arms:
            ref += a
        ref /= refs
        d = (test - ref).mean(axis=2)
        mean, sd = d.mean(axis=1), d.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (mean + M) / (sd / math.sqrt(CASES))
        t = np.where(sd > 0, t, np.where(mean > -M, np.inf, -np.inf))
        out.append(t)
    return np.concatenate(out)



def spread(sims, q, head):
    for refs in (1, 2, 3):
        print(f"\nreference = mean of {refs} arm(s)")
        print(f"pd_ref  rho   k | pd_test | error {head} | power0 {head}")
        for pd_ref in (0.01, 0.02, 0.026, 0.03, 0.04, 0.045, 0.06, 0.08):
            mu = pd_ref / 2
            for rho in (0.0, 0.15, 0.3):
                for k in (1, 2):
                    rng = np.random.default_rng(int(pd_ref * 1e4) * 100 + int(rho * 100) + k + refs * 7)
                    pd_test = (1 + k) * mu     # test vs one reference arm, at difference 0, approx.
                    k1 = solve_k1(mu, rho, k)
                    t0 = tstats(rng, mu, rho, k, solve_k1(mu, rho, k, 0.0), refs, sims)
                    pw = " ".join(f"{(t0 > q[c]).mean():6.3f}" for c in LEVELS)
                    if k1 is None or pd_test < M:
                        er = " ".join("     -" for _ in LEVELS)
                    else:
                        te = tstats(rng, mu, rho, k, k1, refs, sims)
                        er = " ".join(f"{(te > q[c]).mean():6.3f}" for c in LEVELS)
                    print(f"{pd_ref:5.3f}  {rho:4.2f}  {k} |  {pd_test:5.3f}  |       {er} |        {pw}")
                    sys.stdout.flush()


def tight(sims, q, head):
    for refs in (2, 3):
        print(f"\nreference = mean of {refs} arm(s)")
        print(f"pd_ref pd_test rho    k | error {head} | power0 {head}")
        for pd_ref in (0.002, 0.01, 0.02):
            mu = pd_ref / 2
            for pd_test in (0.03, 0.045, 0.06, 0.08):
                k = pd_test / mu - 1
                for rho in (0.0, 0.15, 0.3):
                    rng = np.random.default_rng(int(pd_ref * 1e4) * 1000 + int(pd_test * 1e3) * 10 + int(rho * 10) + refs)
                    k1 = solve_k1(mu, rho, k)
                    t0 = tstats(rng, mu, rho, k, solve_k1(mu, rho, k, 0.0), refs, sims)
                    te = tstats(rng, mu, rho, k, k1, refs, sims) if k1 else None
                    er = " ".join(f"{(te > q[c]).mean():6.3f}" for c in LEVELS) if te is not None else "-"
                    pw = " ".join(f"{(t0 > q[c]).mean():6.3f}" for c in LEVELS)
                    print(f"{pd_ref:5.3f} {pd_test:6.3f} {rho:4.2f} {k:5.1f} |       {er} |        {pw}")
                    sys.stdout.flush()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sims", type=int, nargs="?", default=4000)
    p.add_argument("--table", choices=("spread", "tight", "both"), default="both")
    args = p.parse_args(argv)
    q = {c: stats.t.ppf(c, CASES - 1) for c in LEVELS}
    head = " ".join(f"{c * 100:6.2f}" for c in LEVELS)
    print(f"{args.sims} sims/point; acc {ACC}; MCSE near 0.025 = {math.sqrt(0.025 * 0.975 / args.sims):.4f}")
    if args.table in ("spread", "both"):
        spread(args.sims, q, head)
    if args.table in ("tight", "both"):
        tight(args.sims, q, head)
    return 0


if __name__ == "__main__":
    sys.exit(main())
