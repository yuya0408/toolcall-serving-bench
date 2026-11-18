#!/usr/bin/env bash
# The prefill-kernel check (docs/prefill-kernel-check.md): set up vLLM only, then
# time each configuration with scripts/prefill_bench.py, twice in reversed order.
#
#     scripts/gpu/local.sh prefill <ip>     # from WSL; starts this script detached
#
# Lighter than setup.sh: no SGLang, TensorRT-LLM or llama.cpp, and no ModelOpt
# checkpoints. Writes RUNNING / DONE / FAILED in ~/out like run_cells.py, so
# `local.sh collect` and `status` work unchanged.
set -euo pipefail

REPO=~/toolcall-serving-bench
OUT=~/out
mkdir -p "$OUT/prefill" "$OUT/checks"
rm -f "$OUT/DONE" "$OUT/FAILED"
touch "$OUT/RUNNING"
trap 'echo "prefill_bench.sh line $LINENO: $BASH_COMMAND" > "$OUT/FAILED"; rm -f "$OUT/RUNNING"' ERR
source "$REPO/scripts/gpu/pins.sh"
step() { echo "== $(date -u +%H:%M:%S) $*"; }

step "GPU"
nvidia-smi | tee "$OUT/checks/nvidia-smi.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
  --format=csv -l 5 > "$OUT/gpu.csv" &
SMI=$!

step "vLLM environment"
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --allow-existing --seed --python 3.12 ~/venv
uv pip install --python ~/venv/bin/python "vllm==$VLLM_VERSION" "transformers==$TRANSFORMERS_VERSION" \
  --torch-backend=auto
uv pip freeze --python ~/venv/bin/python > "$OUT/checks/pip-freeze-venv.txt"
~/venv/bin/python -c "import vllm, sys; sys.exit(vllm.__version__ != '$VLLM_VERSION')" ||
  { echo "vllm is not $VLLM_VERSION" > "$OUT/FAILED"; rm -f "$OUT/RUNNING"; exit 1; }

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
  "fp8-tensor|{\"model\": \"Qwen/Qwen3-8B\", \"revision\": \"$QWEN3_8B_REVISION\", \"quantization\": \"fp8\"}"
  "awq|{\"model\": \"Qwen/Qwen3-8B-AWQ\", \"revision\": \"$QWEN3_8B_AWQ_REVISION\"}"
)
run_() {
  local round=$1 name=${2%%|*} engine=${2#*|}
  step "round $round: $name"
  ~/venv/bin/python "$REPO/scripts/prefill_bench.py" --name "$name-r$round" --engine "$engine" \
    --out "$OUT/prefill" > "$OUT/prefill/$name-r$round.log" 2>&1
  grep -E "Selected|Using .* for" "$OUT/prefill/$name-r$round.log" > "$OUT/prefill/$name-r$round.kernels.txt" || true
  tail -n 1 "$OUT/prefill/$name-r$round.log"
}
# Round 2 runs in reverse, so that a drift in the GPU's clock over the run
# shows as a difference between the rounds rather than between configurations.
for c in "${CONFIGS[@]}"; do run_ 1 "$c"; done
for ((i = ${#CONFIGS[@]} - 1; i >= 0; i--)); do run_ 2 "${CONFIGS[$i]}"; done

kill "$SMI" 2>/dev/null || true
rm -f "$OUT/RUNNING"
date -u +"prefill check finished %Y-%m-%dT%H:%M:%SZ" > "$OUT/DONE"
