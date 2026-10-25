"""Rewrite a latency manifest's measured_stability from its raw log.

Cells run before 2026-10-25 cut the measured requests into windows in
completion order, which put each cycle's slow requests late (ADR-0003,
2026-10-25). The raw log keeps every request's send and finish time, so the
send-order windows can be computed without running the cell again. Nothing
else in the manifest changes: the verdicts use all measured requests, not
the windows.

    python scripts/recompute_stability.py runs/<dir>/lat-*/latency-*Z.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_bench.latency import STABILITY_PCT, STABLE_WINDOWS, is_stable, measured_windows  # noqa: E402
from eval_bench.runner import _MANIFEST_SCHEMA  # noqa: E402


def recompute(manifest_path: Path) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = manifest_path.with_name(manifest["run_id"] + ".raw.jsonl")
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
    by_index = {r["index"]: r for r in rows}
    measured = sorted((r for r in rows if r["phase"] == "measured"), key=lambda r: r["index"])
    old = manifest["results"]["measured_stability"]
    if old.get("window_order") == "sent":
        return old, old   # already send order; nothing to redo
    window_size = old["window_requests"]
    after = measured[-1]["index"] + 1
    # Same end as the driver: the send after the measured span, or at
    # concurrency 1 (nothing sent after it) the last measured completion.
    end = by_index[after]["sent_at_s"] if after in by_index else max(r["done_at_s"] for r in measured)
    windows = measured_windows([r["sent_at_s"] for r in measured],
                               [r["total_ms"] / 1000 for r in measured], end, window_size)
    new = {
        "window_requests": window_size,
        "windows": windows,
        "window_order": "sent",
        "recomputed_from_raw": True,
        "stability_pct": STABILITY_PCT,
        "stable": is_stable(windows, last=len(windows)) if len(windows) >= STABLE_WINDOWS else None,
    }
    manifest["results"]["measured_stability"] = new
    jsonschema.validate(instance=manifest, schema=_MANIFEST_SCHEMA)
    # Written as eval_bench.runner.write_manifest writes it.
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return old, new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifests", nargs="+", type=Path)
    for path in parser.parse_args().manifests:
        old, new = recompute(path)
        print(f"{path.parent.name}: stable {old['stable']} -> {new['stable']}")


if __name__ == "__main__":
    main()
