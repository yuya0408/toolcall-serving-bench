"""Tool-set staging: tool count as an experiment variable.

The same cases are run under tool sets of increasing size to produce a
selection-accuracy degradation curve ("how many tools can a small local model
handle?"). Real ERP agents carry dozens of tools, so a number measured against
3 tools does not transfer to production.

Construction rule for a stage of size N, per case:
  1. the correct tool (always present; for negative cases, no anchor)
  2. the case's declared distractor_tools (the deliberately confusable ones)
  3. same-module tools (near-duplicate functions -- the real source of confusion)
  4. other-module tools, to pad out to N

Ordering is shuffled with a fixed seed so position bias does not masquerade as
selection ability, and so runs are reproducible.
"""
from __future__ import annotations
import random
from typing import Any


def build_tool_set(case: dict[str, Any],
                   tools: dict[str, dict[str, Any]],
                   size: int,
                   seed: int = 0) -> list[str]:
    """Return the tool names to present for this case at this stage size."""
    if size > len(tools):
        raise ValueError(f"stage size {size} exceeds available tools ({len(tools)})")

    # A stage must be large enough to hold the correct tool plus every declared
    # distractor. Silently dropping distractors would make the small stages
    # easier than declared and quietly flatten the degradation curve -- the very
    # effect this experiment is meant to measure. Fail loudly instead.
    correct_n = 0 if case.get("expected_tool") is None else 1
    needed = correct_n + len([d for d in case.get("distractor_tools", []) if d in tools])
    if size < needed:
        raise ValueError(
            f"{case['id']}: stage size {size} cannot hold correct tool + "
            f"{needed - correct_n} distractors (needs >= {needed})"
        )

    chosen: list[str] = []
    correct = case.get("expected_tool")
    if correct is not None:
        if correct not in tools:
            raise ValueError(f"{case['id']}: unknown expected_tool {correct!r}")
        chosen.append(correct)

    for d in case.get("distractor_tools", []):
        if d in tools and d not in chosen and len(chosen) < size:
            chosen.append(d)

    rng = random.Random(f"{seed}:{case['id']}:{size}")

    if correct is not None:
        same_module = [n for n, t in tools.items()
                       if t.get("module") == tools[correct].get("module") and n not in chosen]
        rng.shuffle(same_module)
        for n in same_module:
            if len(chosen) >= size:
                break
            chosen.append(n)

    rest = [n for n in tools if n not in chosen]
    rng.shuffle(rest)
    for n in rest:
        if len(chosen) >= size:
            break
        chosen.append(n)

    # Shuffle final order so the correct tool is not always first.
    rng.shuffle(chosen)
    return chosen[:size]


def default_stages(tools: dict[str, Any], declared: list[int] | None = None) -> list[int]:
    """Clamp declared stage sizes to what the tool catalogue can actually support."""
    declared = declared or [5, 10, 20, 40]
    n = len(tools)
    return [s for s in declared if s <= n] + ([n] if n not in declared and n > 3 else [])
