"""Report layer: model x metric matrix, with mandatory caveats.

Every reported number should carry selection-set context (tool count,
distractor count) so a bare '95% selection' can't be read out of context.
Also emits per-provenance breakdowns so readers can exclude generated cases.
"""
from __future__ import annotations
from eval_bench.scorers.separated import CaseScore


def aggregate(scores: list[CaseScore]) -> dict[str, float]:
    def rate(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")
    return {
        "selection":    rate([s.selection for s in scores]),
        "arg_exact":    rate([s.arg_exact for s in scores]),
        "arg_semantic": rate([s.arg_semantic for s in scores]),
        "e2e":          rate([s.e2e for s in scores]),
        "n":            len(scores),
    }
