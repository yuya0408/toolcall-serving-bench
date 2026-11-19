#!/usr/bin/env bash
# The prefill-kernel check (docs/prefill-kernel-check.md): set up vLLM only, then
# time each configuration with scripts/prefill_bench.py, twice in reversed order.
#
#     scripts/gpu/local.sh prefill <ip>     # from WSL; starts this script detached
#
# Lighter than setup.sh: no SGLang, TensorRT-LLM or llama.cpp, and no ModelOpt
# checkpoints. Writes RUNNING / DONE / FAILED in ~/out like run_cells.py, so
# `local.sh collect` and `status` work unchanged: collect waits while RUNNING
# exists, so every way out of this script removes it (the EXIT trap).
# -E: functions inherit the ERR trap (without it a failure inside run_ wrote no FAILED).
set -eEuo pipefail

REPO=~/toolcall-serving-bench
OUT=~/out
mkdir -p "$OUT/prefill" "$OUT/checks"
rm -f "$OUT/DONE" "$OUT/FAILED" "$OUT/prefill/failed.txt"
touch "$OUT/RUNNING"
SMI=
trap 'echo "prefill_bench.sh line $LINENO: $BASH_COMMAND" > "$OUT/FAILED"' ERR
finish_() {
  [ -n "$SMI" ] && kill "$SMI" 2>/dev/null
  [ -f "$OUT/DONE" ] || [ -f "$OUT/FAILED" ] || echo "prefill_bench.sh stopped before the end" > "$OUT/FAILED"
  rm -f "$OUT/RUNNING"
}
trap finish_ EXIT
source "$REPO/scripts/gpu/pins.sh"
step() { echo "== $(date -u +%H:%M:%S) $*"; }

step "GPU"
nvidia-smi | tee "$OUT/checks/nvidia-smi.txt"
# The power cap (72 W on L4) can lower the clock during long prefills; the
# reasons field says when. Older drivers call it clocks_throttle_reasons.
QUERY=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,clocks_event_reasons.active
nvidia-smi --query-gpu=$QUERY --format=csv > /dev/null 2>&1 || QUERY=${QUERY/clocks_event_reasons/clocks_throttle_reasons}
nvidia-smi --query-gpu=$QUERY --format=csv -lms 500 > "$OUT/gpu.csv" &
SMI=$!

step "vLLM environment"
# ~/venv/bin first: FlashInfer's JIT (vLLM's sampler) runs the venv's ninja, as in run_cells.py.
export PATH="$HOME/venv/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
# Models on the instance store: bf16 (15.3 GiB) is over 90% of the free RAM, so vLLM
# turns off its prefetch and reads by mmap page faults, which the gp3 root volume
# served at ~5.6 MB/s (2026-11-18, ~7 min per shard).
if [ -d /opt/dlami/nvme ]; then export HF_HOME=/opt/dlami/nvme/hf; fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --allow-existing --seed --python 3.12 ~/venv
uv pip install --python ~/venv/bin/python "vllm==$VLLM_VERSION" "transformers==$TRANSFORMERS_VERSION" \
  --torch-backend=auto
uv pip freeze --python ~/venv/bin/python > "$OUT/checks/pip-freeze-venv.txt"
~/venv/bin/python -c "import vllm, sys; sys.exit(vllm.__version__ != '$VLLM_VERSION')"
# Set, it would make Marlin try W8A8, which the FP8 Marlin path does not support.
if [ -n "${VLLM_MARLIN_INPUT_DTYPE:-}" ]; then
  echo "VLLM_MARLIN_INPUT_DTYPE is set; unset it and run again" > "$OUT/FAILED"; exit 1
fi

step "models (pinned revisions)"
~/venv/bin/python - "$QWEN3_8B_REVISION" "$QWEN3_8B_FP8_REVISION" "$QWEN3_8B_AWQ_REVISION" <<'EOF'
import sys
from huggingface_hub import snapshot_download
for repo, rev in zip(("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-FP8", "Qwen/Qwen3-8B-AWQ"), sys.argv[1:]):
    print(repo, snapshot_download(repo, revision=rev))
EOF

# name | vllm.LLM arguments. The expected linear kernel on L4 (sm_89) is in
# docs/prefill-kernel-check.md; the one vLLM picked is read from its log below.
CONFIGS=(
  "bf16|{\"model\": \"Qwen/Qwen3-8B\", \"revision\": \"$QWEN3_8B_REVISION\"}"
  "fp8-block|{\"model\": \"Qwen/Qwen3-8B-FP8\", \"revision\": \"$QWEN3_8B_FP8_REVISION\"}"
  "fp8-block-triton|{\"model\": \"Qwen/Qwen3-8B-FP8\", \"revision\": \"$QWEN3_8B_FP8_REVISION\", \"linear_backend\": \"triton\"}"
  "fp8-tensor|{\"model\": \"Qwen/Qwen3-8B\", \"revision\": \"$QWEN3_8B_REVISION\", \"quantization\": \"fp8_per_tensor\"}"
  "awq|{\"model\": \"Qwen/Qwen3-8B-AWQ\", \"revision\": \"$QWEN3_8B_AWQ_REVISION\"}"
)
# One configuration failing is recorded and the rest still run.
run_() {
  local round=$1 name=${2%%|*} engine=${2#*|}
  step "round $round: $name"
  if ! ~/venv/bin/python "$REPO/scripts/prefill_bench.py" --name "$name-r$round" --engine "$engine" \
      --out "$OUT/prefill" > "$OUT/prefill/$name-r$round.log" 2>&1; then
    echo "$name-r$round" >> "$OUT/prefill/failed.txt"
    step "round $round: $name FAILED (see prefill/$name-r$round.log)"
  fi
  grep -E "Selected|Using .* for|W8A8 Block" "$OUT/prefill/$name-r$round.log" \
    > "$OUT/prefill/$name-r$round.kernels.txt" || true
  tail -n 1 "$OUT/prefill/$name-r$round.log"
}
# Round 2 runs in reverse, so that a drift in the GPU's clock over the run
# shows as a difference between the rounds rather than between configurations.
for c in "${CONFIGS[@]}"; do run_ 1 "$c"; done
for ((i = ${#CONFIGS[@]} - 1; i >= 0; i--)); do run_ 2 "${CONFIGS[$i]}"; done

failed=
[ -f "$OUT/prefill/failed.txt" ] && failed=$(tr '\n' ' ' < "$OUT/prefill/failed.txt")
date -u +"prefill check finished %Y-%m-%dT%H:%M:%SZ; failed: ${failed:-none}" > "$OUT/DONE"
