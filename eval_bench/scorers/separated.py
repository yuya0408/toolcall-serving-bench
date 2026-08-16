"""Separated scoring.

Design rule: the three primary metrics are computed independently and NEVER
folded together. E2E is a derived convenience metric only.

  - selection:   did the model pick the right tool (incl. correctly picking
                 "no tool" for negative cases)?
  - arg_exact:   for exact-scored fields (enums/numbers/dates), all match?
  - arg_semantic: for free-text fields, similarity/judge above threshold?

The semantic judge is injected (dependency), not hard-coded, so it can be an
embedding-similarity check or an LLM-as-Judge without touching this module.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# A semantic judge takes (expected_text, predicted_text) -> pass/fail (bool).
SemanticJudge = Callable[[str, str], bool]


@dataclass
class Prediction:
    """What a model produced for one case."""
    tool: Optional[str]          # None means the model declined to call a tool
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseScore:
    case_id: str
    selection: bool
    arg_exact: Optional[bool]     # None when not applicable (e.g. negative case, or no exact fields)
    arg_semantic: Optional[bool]  # None when not applicable
    # E2E is derived, not a primary metric. Kept out of the primary trio on purpose.
    e2e: bool


def score_selection(expected_tool: Optional[str], pred_tool: Optional[str]) -> bool:
    # Correctly declining on a negative case counts as a selection success.
    return expected_tool == pred_tool


def score_arg_exact(expected_exact: dict[str, Any], pred_args: dict[str, Any]) -> Optional[bool]:
    if not expected_exact:
        return None
    return all(pred_args.get(k) == v for k, v in expected_exact.items())


def score_arg_semantic(
    expected_semantic: dict[str, Any],
    pred_args: dict[str, Any],
    judge: SemanticJudge,
) -> Optional[bool]:
    if not expected_semantic:
        return None
    return all(
        judge(str(v), str(pred_args.get(k, ""))) for k, v in expected_semantic.items()
    )


def score_case(case: dict[str, Any], pred: Prediction, judge: SemanticJudge) -> CaseScore:
    sel = score_selection(case["expected_tool"], pred.tool)

    # Argument scoring is only meaningful when the right tool was selected.
    # If selection failed, arg metrics are N/A rather than a forced failure,
    # so that argument-extraction ability is not conflated with selection.
    if sel and not case["negative"]:
        exact = score_arg_exact(case["expected_args"].get("exact", {}), pred.args)
        semantic = score_arg_semantic(case["expected_args"].get("semantic", {}), pred.args, judge)
    else:
        exact = None
        semantic = None

    # Derived only. E2E = selection AND every applicable arg metric passed.
    e2e = sel and (exact in (True, None)) and (semantic in (True, None)) and not (
        # a negative case is E2E-correct exactly when selection was correct
        False
    )

    return CaseScore(
        case_id=case["id"],
        selection=sel,
        arg_exact=exact,
        arg_semantic=semantic,
        e2e=e2e,
    )
