"""Semantic judges (injected into the scorer).

Primary judge for reporting = embedding similarity with a fixed, *calibrated*
threshold (reproducible). LLM-as-Judge is provided only as an audit judge to
cross-check the embedding judge on the seed set; it is NOT the reporting judge.

Both satisfy the SemanticJudge signature: (expected, predicted) -> bool.
"""
from __future__ import annotations
from typing import Callable, Protocol


class Embedder(Protocol):
    def encode(self, text: str) -> list[float]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def embedding_judge(embedder: Embedder, threshold: float) -> Callable[[str, str], bool]:
    """Reporting judge. Deterministic given a fixed embedder + threshold."""
    def judge(expected: str, predicted: str) -> bool:
        if not predicted:
            return False
        return _cosine(embedder.encode(expected), embedder.encode(predicted)) >= threshold
    return judge


def calibrate_threshold(embedder: Embedder,
                        positives: list[tuple[str, str]],
                        negatives: list[tuple[str, str]]) -> dict[str, float]:
    """Pick the threshold that best separates hand-labeled positive/negative
    text pairs (from the seed set). Returns the chosen threshold plus the
    accuracy at +/-0.05 so a report can show sensitivity, not a bare cut.
    """
    def acc(t: float) -> float:
        ok = sum(1 for e, p in positives if _cosine(embedder.encode(e), embedder.encode(p)) >= t)
        ok += sum(1 for e, p in negatives if _cosine(embedder.encode(e), embedder.encode(p)) < t)
        return ok / (len(positives) + len(negatives))

    grid = [i / 100 for i in range(50, 96)]
    best = max(grid, key=acc)
    return {
        "threshold": best,
        "acc_at_threshold": acc(best),
        "acc_minus_0.05": acc(max(0.0, best - 0.05)),
        "acc_plus_0.05": acc(min(1.0, best + 0.05)),
    }
