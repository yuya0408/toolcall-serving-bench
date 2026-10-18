#!/usr/bin/env bash
# One-time setup on the GPU instance (docs/gpu-runbook.md). Run from the repo
# copy that scripts/gpu/local.sh pushed to ~/toolcall-serving-bench:
#
#     bash scripts/gpu/setup.sh            # ~/out/setup.log has the full output
#
# Everything a manifest or a later reader needs is written to ~/out as soon as
# it is known (the instance and its disk disappear at the auto-shutdown):
#   ~/out/versions.json     engine versions, llama.cpp commit, model revisions
#   ~/out/checks/*.txt      the parser list and version checks (runbook step 4)
set -euo pipefail

OUT=~/out
MODELS=~/models
REPO=~/toolcall-serving-bench
mkdir -p "$OUT/checks" "$MODELS"
rm -f "$OUT/SETUP_DONE" "$OUT/SETUP_FAILED"
# A marker local.sh collect stops on; the failing command is the last lines of ~/out/setup.log.
trap 'echo "setup.sh line $LINENO: $BASH_COMMAND" > "$OUT/SETUP_FAILED"' ERR

# Each step's start time, so a later apply can be planned within the 6-hour shutdown.
step() { echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

# Monitors for the whole life of the instance (they outlive this script; a rerun
# does not start a second copy). status.jsonl's timestamps say which cell ran when.
#   gpu.csv: VRAM, utilization, power, temperature and SM clock every 5 s -- a clock
#            that drops under sustained load would make later latency cells look worse
#   vmstat.log: host memory every 5 s -- 16 GB, about the size of the bf16 weights
if ! pgrep -f "nvidia-smi --query-gpu=timestamp" >/dev/null; then
  # Why the clock dropped, if it did; the field was renamed (clocks_event_reasons in
  # newer drivers), and a name the driver rejects would stop the whole log.
  GPU_FIELDS=timestamp,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm
  for reason in clocks_event_reasons.active clocks_throttle_reasons.active; do
    if nvidia-smi --query-gpu="$GPU_FIELDS,$reason" --format=csv,noheader >/dev/null 2>&1; then
      GPU_FIELDS="$GPU_FIELDS,$reason"; break
    fi
  done
  nohup nvidia-smi --query-gpu="$GPU_FIELDS" --format=csv -l 5 > "$OUT/gpu.csv" 2>&1 < /dev/null &
fi
if ! pgrep -f "vmstat -t -S M 5" >/dev/null; then
  nohup vmstat -t -S M 5 > "$OUT/vmstat.log" 2>&1 < /dev/null &
fi

export PATH="/usr/local/cuda/bin:$PATH"   # the DLAMI installs nvcc there, not always on PATH
step "AMI prerequisites"
nvidia-smi | tee "$OUT/checks/nvidia-smi.txt"
nvcc --version | tee "$OUT/checks/nvcc.txt"   # no nvcc -> no CUDA build of llama.cpp; stop here
{ shutdown --show 2>/dev/null || cat /run/systemd/shutdown/scheduled; } 2>&1 | tee "$OUT/checks/shutdown.txt" || true   # --show needs systemd >= 250

step "system packages"
sudo apt-get update -q
sudo apt-get install -y -q build-essential cmake git rsync

step "temporary swap for the CUDA build and the GGUF conversion"
# 16 GB of RAM is about the size of the bf16 weights. Nothing here is expected to
# need more, but a build or conversion killed for memory would cost the whole
# setup again; with swap it only slows down. It is removed at the end of setup,
# so no cell is ever measured with swap on (run_cells.py refuses to start otherwise).
if ! grep -q /swapfile-setup /proc/swaps; then
  sudo fallocate -l 16G /swapfile-setup
  sudo chmod 600 /swapfile-setup
  sudo mkswap /swapfile-setup >/dev/null
  sudo swapon /swapfile-setup
fi

step "uv and the engine environment"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv --allow-existing --python 3.12 ~/venv
# vLLM brings torch and transformers; accelerate is for transformers_hf's device_map.
uv pip install --python ~/venv/bin/python vllm accelerate -r "$REPO/requirements.txt" --torch-backend=auto

step "llama.cpp (CUDA, L4 = sm_89)"
if [ ! -d ~/llama.cpp ]; then
  git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
fi
git -C ~/llama.cpp rev-parse HEAD > "$OUT/llama_cpp_commit.txt"   # never recoverable after destroy
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build ~/llama.cpp/build --config Release -j "$(nproc)" --target llama-server

step "model weights (Hugging Face cache)"
~/venv/bin/python - <<'EOF' > "$OUT/checks/model_paths.txt"
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-AWQ"):
    print(repo, snapshot_download(repo))
EOF
cat "$OUT/checks/model_paths.txt"

step "BF16 GGUF from the same safetensors (docs/engine-performance-plan.md)"
# A separate environment: the converter's requirements pin a CPU torch that
# would replace vLLM's CUDA build.
uv venv --allow-existing --python 3.12 ~/venv-convert
uv pip install --python ~/venv-convert/bin/python -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
SNAPSHOT=$(awk '$1 == "Qwen/Qwen3-8B" {print $2}' "$OUT/checks/model_paths.txt")
if [ ! -f "$MODELS/Qwen3-8B-bf16.gguf" ]; then
  ~/venv-convert/bin/python ~/llama.cpp/convert_hf_to_gguf.py "$SNAPSHOT" \
    --outtype bf16 --outfile "$MODELS/Qwen3-8B-bf16.gguf"
fi

step "version checks (runbook step 4)"
~/venv/bin/vllm serve --help=all 2>/dev/null | grep -A3 -i "tool-call-parser" > "$OUT/checks/vllm_tool_call_parser.txt" || \
  ~/venv/bin/vllm serve --help 2>&1 | grep -A3 -i "tool-call-parser" > "$OUT/checks/vllm_tool_call_parser.txt" || true
~/venv/bin/python - "$OUT" <<'EOF'
import json, pathlib, sys
from importlib.metadata import version
out = pathlib.Path(sys.argv[1])
revisions = {}
for line in (out / "checks" / "model_paths.txt").read_text().splitlines():
    repo, path = line.split()
    revisions[repo] = pathlib.Path(path).name   # the snapshot directory is the commit hash
versions = {
    "vllm": version("vllm"), "transformers": version("transformers"), "torch": version("torch"),
    "llama_cpp_commit": (out / "llama_cpp_commit.txt").read_text().strip(),
    "model_revisions": revisions,
}
(out / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
print(json.dumps(versions, indent=2))
EOF
grep -c hermes "$OUT/checks/vllm_tool_call_parser.txt" >/dev/null && echo "hermes parser: listed" || echo "hermes parser: NOT FOUND -- check $OUT/checks/vllm_tool_call_parser.txt"
step "does llama.cpp honour seed (runbook step 4c)"
~/venv/bin/python "$REPO/scripts/gpu/check_llama_seed.py" || echo "llama seed check failed -- see $OUT/checks/"
step "removing the temporary swap"
sudo swapoff /swapfile-setup
sudo rm -f /swapfile-setup
df -h / ~ > "$OUT/checks/disk.txt"; cat "$OUT/checks/disk.txt"
step "setup done"
touch "$OUT/SETUP_DONE"
