"""Orchestrates one manifest's worth of measurement.

Wires an adapter+decoder pair to the dataset (eval_bench/pipeline.py), runs
the accuracy sweep (eval_bench/repetition.py), and assembles+validates a
schema/run_manifest.schema.json-shaped manifest. Pure Python: EchoAdapter
exercises every line of this file with no GPU/engine involved -- connecting
a real backend only swaps which ModelAdapter gets passed in. Terraform/the
GPU instance lifecycle is a separate, later concern (docs/engine-performance-
plan.md's コスト管理 section: apply -> run -> destroy, user-executed).
"""
from __future__ import annotations
import json
import pathlib
import re
from datetime import datetime, timezone
from typing import Any, Optional

import jsonschema

from eval_bench.dataset.tools import load_tools
from eval_bench.repetition import (
    PredictFn,
    aggregate_repeated,
    run_repeated_case,
    run_repeated_cases_concurrent,
)
from eval_bench.scorers.separated import SemanticJudge

_SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "schema" / "run_manifest.schema.json"
_MANIFEST_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

# The original 40-tool catalog, loaded once as the reference set for the
# stage=10 invariant below (schema/tools.json's own default path).
_ORIGINAL_40_TOOL_NAMES = frozenset(load_tools().keys())

_COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{7,40}$")

_VLLM_GPU_BLOCKS_RE = re.compile(r"#\s*GPU blocks:\s*(\d+)")
_VLLM_MAX_CONCURRENCY_RE = re.compile(r"[Mm]aximum concurrency for \d+ tokens per request:\s*([\d.]+)")


def parse_vllm_diagnostics(startup_log: str) -> dict[str, Any]:
    """Best-effort parse of vLLM's startup log for '# GPU blocks' and
    'Maximum concurrency for N tokens per request'.

    The exact wording is UNVERIFIED against a real vLLM version -- docs/
    engine-performance-plan.md flags both the version floor and this log
    format as boot-time confirmation items. If these regexes don't match
    the real log, fix them here; _require_vllm_diagnostics_if_needed below
    (the actual safety guard) doesn't change either way -- it only cares
    about the parsed dict's shape, not how it was produced.
    """
    blocks_match = _VLLM_GPU_BLOCKS_RE.search(startup_log)
    concurrency_match = _VLLM_MAX_CONCURRENCY_RE.search(startup_log)
    if blocks_match is None or concurrency_match is None:
        raise ValueError(
            "could not find '# GPU blocks' and/or 'Maximum concurrency for N tokens "
            "per request' in the given vLLM startup log -- verify these regexes against "
            "the actual log wording for the vLLM version running (unverified, see "
            "docs/engine-performance-plan.md's 'depends on vllm serve --help at boot')"
        )
    return {
        "gpu_blocks_reported": int(blocks_match.group(1)),
        "observed_max_concurrency": float(concurrency_match.group(1)),
    }


def _require_vllm_diagnostics_if_needed(cell: dict[str, Any], results: dict[str, Any]) -> None:
    """vLLM does not OOM when KV capacity is exceeded -- it queues excess
    requests -- so a concurrency>1 vllm manifest without vllm_diagnostics,
    or with observed_max_concurrency below what was requested, cannot be
    trusted to reflect true concurrent batching rather than serialized
    queueing delay reported as throughput/accuracy-under-load
    (schema/run_manifest.schema.json's vllm_diagnostics description;
    docs/engine-performance-plan.md's '要実装'). Refuses to build the
    manifest rather than persisting data that can't be trusted -- this is
    "the execution harness['s] responsib[ility]" the schema explicitly
    delegates, since the schema itself can't check an inequality between
    two of its own fields.
    """
    if cell.get("engine") != "vllm" or cell.get("concurrency", 1) <= 1:
        return
    diagnostics = results.get("vllm_diagnostics")
    if diagnostics is None:
        raise ValueError(
            f"cell.concurrency={cell['concurrency']} with engine=vllm requires "
            "results['vllm_diagnostics'] (gpu_blocks_reported, observed_max_concurrency) -- "
            "refusing to write a manifest that can't be trusted to reflect true concurrent "
            "batching rather than serialized queueing delay"
        )
    observed = diagnostics["observed_max_concurrency"]
    requested = cell["concurrency"]
    if observed < requested:
        raise ValueError(
            f"observed_max_concurrency ({observed}) < cell.concurrency ({requested}): "
            "this run measured queueing delay, not concurrent batching -- discard it, "
            "do not report it (docs/engine-performance-plan.md '要実装')"
        )


def _validate_cell_invariants(pass_type: str, cell: dict[str, Any],
                              tools: dict[str, dict[str, Any]]) -> None:
    """Three rules docs/engine-performance-plan.md states but nothing in
    code enforced until now. Runs before any model call (inside
    run_accuracy_cell, not just at manifest-build time) so a misconfigured
    cell fails before spending GPU time, not after.
    """
    tool_stage = cell.get("tool_stage")

    if tool_stage == 10 and frozenset(tools.keys()) != _ORIGINAL_40_TOOL_NAMES:
        raise ValueError(
            "tool_stage=10 must use the original 40-tool catalog (schema/tools.json) -- "
            "passing the extended catalog changes the padding candidate pool and breaks "
            "the Article-1-overlay property (docs/engine-performance-plan.md)"
        )

    if pass_type == "accuracy_under_load" and tool_stage != 10:
        raise ValueError(
            "accuracy_under_load is pinned to tool_stage=10 (2026-09-11 decision, "
            "docs/engine-performance-plan.md '並行数の具体的な水準'): per-trial tool "
            "shuffling collapses prefix sharing beyond the system prompt, so higher "
            "stages don't fit AWQ's concurrency=16 headroom"
        )

    if cell.get("engine") == "llama_cpp":
        version = cell.get("engine_version", "")
        if not _COMMIT_HASH_RE.match(version):
            raise ValueError(
                f"llama_cpp's engine_version must be a git commit hash (7-40 hex chars), "
                f"got {version!r}. docs/engine-performance-plan.md requires commit-hash "
                "granularity (llama.cpp's version tags are too coarse to pin quantization/"
                "kernel behavior), and this cannot be reconstructed once the GPU instance "
                "that built it is destroyed -- it must be supplied, never defaulted."
            )


def run_accuracy_cell(
    cases: list[dict[str, Any]],
    tools: dict[str, dict[str, Any]],
    cell: dict[str, Any],
    pass_type: str,
    predict: PredictFn,
    judge: SemanticJudge,
    n: int,
) -> dict[str, Any]:
    """Run every case x n trials at cell['tool_stage'] and aggregate -- the
    accuracy_results_base shape both 'accuracy' and 'accuracy_under_load'
    manifests share (schema/run_manifest.schema.json's $defs).

    cell['concurrency'] > 1 switches to run_repeated_cases_concurrent (real
    in-flight parallelism) instead of the serialized per-case loop -- see
    that function's docstring for why this is required, not optional, for
    the accuracy_under_load probe to measure what it claims to.
    """
    _validate_cell_invariants(pass_type, cell, tools)
    concurrency = cell.get("concurrency", 1)
    if concurrency > 1:
        results = run_repeated_cases_concurrent(
            cases, tools, cell["tool_stage"], predict, judge, n=n, max_workers=concurrency,
        )
    else:
        results = [
            run_repeated_case(c, tools, cell["tool_stage"], predict, judge, n=n)
            for c in cases
        ]
    return aggregate_repeated(results)


def build_manifest(
    run_id: str,
    pass_type: str,
    cell: dict[str, Any],
    hardware: dict[str, Any],
    results: dict[str, Any],
    repetition: Optional[dict[str, Any]] = None,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble a manifest and validate it against
    schema/run_manifest.schema.json before returning it. A manifest that
    doesn't validate is refused here, not discovered later by whoever tries
    to read it back.
    """
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass_type": pass_type,
        "cell": cell,
        "hardware": hardware,
        "results": results,
    }
    if repetition is not None:
        manifest["repetition"] = repetition

    _require_vllm_diagnostics_if_needed(cell, results)

    # Normalize to what will actually be written/read as JSON (e.g.
    # aggregate_repeated's selection_ci95 is a Python tuple; jsonschema's
    # default array type check only accepts list, so validating the raw
    # in-memory dict would silently fail every properties check the tuple's
    # $ref touches -- validate exactly what write_manifest will persist).
    manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
    jsonschema.validate(instance=manifest, schema=_MANIFEST_SCHEMA)
    return manifest


def write_manifest(manifest: dict[str, Any], out_dir: pathlib.Path | str) -> pathlib.Path:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest['run_id']}.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
