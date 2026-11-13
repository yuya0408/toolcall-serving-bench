"""scripts/gate_simulation.py must reach the same verdict as the phase-2 gate
itself (noninferiority.pooled_noninferiority), or the error rate it calibrated
the 99.9% level with (docs/adr/0009 5) belongs to a different rule. The full
check (181 error cells x 4,000 datasets, no disagreement) is in docs/adr/0009
A3; this runs a few cells, including the worst one, at both differences."""
from __future__ import annotations
import importlib.util
import pathlib

import pytest

np = pytest.importorskip("numpy")
stats = pytest.importorskip("scipy.stats")

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sim = _load("gate_simulation")
ni = _load("noninferiority")

_KEYS = [(f"c{c:03d}", t) for c in range(sim.CASES) for t in range(sim.TRIALS)]
_SIMS = 100


def _scored(arm):
    return dict(zip(_KEYS, arm.astype(int).reshape(-1).tolist()))


# (references, mu = p_d of a reference / 2, rho, k0, seed); the first two tie for
# the worst simulated error at 99.9% (2.3%)
_CELLS = [(1, 0.013, 0.3, 1, 1), (3, 0.001, 0.15, 79.0, 2), (2, 0.03, 0.0, 2, 3)]


@pytest.mark.parametrize("refs,mu,rho,k0,seed", _CELLS)
@pytest.mark.parametrize("target", [-sim.M, 0.0])
def test_simulated_verdicts_match_the_gate(refs, mu, rho, k0, seed, target):
    k1 = sim.solve_k1(mu, rho, k0, target)
    q = stats.t.ppf(ni.POOLED_CONFIDENCE, sim.CASES - 1)
    t = sim.tstats(np.random.default_rng(seed), mu, rho, k0, k1, refs, _SIMS)
    tq = ni.t_quantile(ni.POOLED_CONFIDENCE, sim.CASES - 1)
    gate, gate_t = [], []
    for test, ref_arms in sim.arms(np.random.default_rng(seed), mu, rho, k0, k1, refs, _SIMS):
        for j in range(test.shape[0]):
            r = ni.pooled_noninferiority([_scored(a[j]) for a in ref_arms], _scored(test[j]))
            gate.append(r["verdict"] == "non-inferior")
            # the gate's t statistic, recovered from its lower bound (mean - tq * se)
            se = (r["selection_diff"] - r["lower_bound"]) / tq
            gate_t.append((r["selection_diff"] + ni.MARGIN) / se if se else t[len(gate_t)])
    assert list(t > q) == gate
    # the statistic itself, so a change too small to flip a verdict here is caught too
    assert gate_t == pytest.approx(list(t), rel=1e-9)
