#!/usr/bin/env bash
# One-time setup on the GPU instance (docs/gpu-runbook.md). Run from the repo
# copy that scripts/gpu/local.sh pushed to ~/toolcall-serving-bench:
#
#     bash scripts/gpu/setup.sh            # ~/out/setup.log has the full output
#     SETUP_LLAMA=0 bash scripts/gpu/setup.sh   # docs/adr/0009: no llama.cpp, no GGUFs
#
# Everything a manifest or a later reader needs is written to ~/out as soon as
# it is known (the instance and its disk disappear at the auto-shutdown):
#   ~/out/versions.json     engine versions, llama.cpp commit, model revisions
#   ~/out/checks/*.txt      the parser list and version checks (runbook step 4)
set -euo pipefail

OUT=~/out
MODELS=~/models
REPO=~/toolcall-serving-bench
# 1: build llama.cpp and the GGUFs (the first stage). The continuous-batching
# stage (docs/adr/0009) measures vLLM, SGLang and TensorRT-LLM only and skips them.
SETUP_LLAMA=${SETUP_LLAMA:-1}
source "$REPO/scripts/gpu/pins.sh"   # the engine versions and model revisions every apply runs
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
# TensorRT-LLM's wheel needs MPI at import time (docs/source/installation/linux.md, v1.2.1).
sudo apt-get install -y -q libopenmpi-dev

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
# --seed puts pip in the venv: vLLM's dev-mode /server_info lists packages with
# `python -m pip list`, and without pip it falls back to `uv pip list`, which
# fails from the server's environment and turns the endpoint into a 500
# (capacity-smoke, 2026-10-24). pip is only read there, never used to install.
uv venv --allow-existing --seed --python 3.12 ~/venv
# vLLM brings torch and transformers; accelerate is for transformers_hf's device_map.
uv pip install --python ~/venv/bin/python "vllm==$VLLM_VERSION" "transformers==$TRANSFORMERS_VERSION" \
  accelerate -r "$REPO/requirements.txt" --torch-backend=auto

step "SGLang in its own environment (docs/adr/0009)"
# SGLang 0.5.21 pins torch 2.13.0 and transformers 5.12.1, unlike the vLLM
# environment; it serves from here, and the client still runs in ~/venv.
uv venv --allow-existing --seed --python 3.12 ~/venv-sglang
uv pip install --python ~/venv-sglang/bin/python "sglang==$SGLANG_VERSION" --torch-backend=auto

step "TensorRT-LLM in its own environment (docs/adr/0009)"
# The documented pip install (v1.2.1 docs/source/installation/linux.md): torch
# 2.9.1 for CUDA 13.0 first, then the wheel from NVIDIA's index with torch held,
# so the resolver cannot swap in the CUDA 12.8 build.
export CUDA_HOME=/usr/local/cuda
uv venv --allow-existing --seed --python 3.12 ~/venv-trtllm
uv pip install --python ~/venv-trtllm/bin/python "torch==$TRTLLM_TORCH_VERSION" torchvision \
  --index-url https://download.pytorch.org/whl/cu130
echo "torch==$(~/venv-trtllm/bin/python -c 'import torch; print(torch.__version__)')" > ~/trtllm-torch-constraint.txt
uv pip install --python ~/venv-trtllm/bin/python "tensorrt_llm==$TRTLLM_VERSION" \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match -c ~/trtllm-torch-constraint.txt
# trtllm-serve --config: iteration statistics, so /metrics reports the KV
# capacity (scripts/gpu/run_cells.py TRTLLM_CONFIG). Observability only.
echo "enable_iter_perf_stats: true" > ~/trtllm-serve.yaml

if [ "$SETUP_LLAMA" = 1 ]; then
step "llama.cpp (CUDA, L4 = sm_89)"
if [ ! -d ~/llama.cpp ]; then
  git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
fi
git -C ~/llama.cpp checkout -q --detach "$LLAMA_CPP_COMMIT"
git -C ~/llama.cpp rev-parse HEAD > "$OUT/llama_cpp_commit.txt"   # never recoverable after destroy
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build ~/llama.cpp/build --config Release -j "$(nproc)" --target llama-server llama-quantize
fi

step "model weights (Hugging Face cache)"
~/venv/bin/python - <<'EOF' > "$OUT/checks/model_paths.txt"
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-8B-FP8"):
    print(repo, snapshot_download(repo))
EOF
cat "$OUT/checks/model_paths.txt"

SNAPSHOT=$(awk '$1 == "Qwen/Qwen3-8B" {print $2}' "$OUT/checks/model_paths.txt")

step "TensorRT-LLM's 4-bit and FP8 from the same safetensors (docs/adr/0009 4)"
# TensorRT-LLM 1.2.1 reads no HF AWQ checkpoint, and its quantization table has
# no FP8 block scaling for Qwen3 or for Ada (docs/source/features/quantization.md),
# so both come from the documented route: ModelOpt's hf_ptq.py on Qwen/Qwen3-8B.
# --kv_cache_qformat none: the KV cache stays unquantized, as for every engine
# (nvidia/Qwen3-8B-FP8 also quantizes the KV cache, so it is not used).
# Calibration is hf_ptq.py's default (512 samples of cnn_dailymail).
if [ ! -d ~/Model-Optimizer ]; then
  git clone -q --branch "$MODELOPT_VERSION" --depth 1 https://github.com/NVIDIA/Model-Optimizer.git ~/Model-Optimizer
fi
uv pip install --python ~/venv-trtllm/bin/python -c ~/trtllm-torch-constraint.txt fire rouge_score datasets
for q in int4_awq:Qwen3-8B-int4-awq-modelopt fp8:Qwen3-8B-fp8-modelopt; do
  format=${q%%:*}; dir="$MODELS/${q#*:}"
  if [ ! -f "$dir/config.json" ]; then
    (cd ~/Model-Optimizer/examples/llm_ptq && ~/venv-trtllm/bin/python hf_ptq.py --pyt_ckpt_path "$SNAPSHOT" \
      --qformat "$format" --kv_cache_qformat none --export_fmt hf --export_path "$dir") \
      2>&1 | tee "$OUT/checks/modelopt-$format.log"
  fi
  cp "$dir"/hf_quant_config.json "$OUT/checks/modelopt-$format-hf_quant_config.json" 2>/dev/null || true
  (cd "$dir" && sha256sum ./*.safetensors) > "$OUT/checks/modelopt-$format-sha256.txt"
done

if [ "$SETUP_LLAMA" = 1 ]; then
step "BF16 GGUF from the same safetensors (docs/engine-performance-plan.md)"
# A separate environment: the converter's requirements pin a CPU torch that
# would replace vLLM's CUDA build.
uv venv --allow-existing --python 3.12 ~/venv-convert
uv pip install --python ~/venv-convert/bin/python -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
if [ ! -f "$MODELS/Qwen3-8B-bf16.gguf" ]; then
  ~/venv-convert/bin/python ~/llama.cpp/convert_hf_to_gguf.py "$SNAPSHOT" \
    --outtype bf16 --outfile "$MODELS/Qwen3-8B-bf16.gguf"
fi

step "Q4_K_M GGUF from the BF16 GGUF (docs/adr/0008)"
# Quantized here with the pinned llama.cpp rather than downloaded, so the chat
# template and every other GGUF field match the BF16 file; only the weights differ.
if [ ! -f "$MODELS/Qwen3-8B-Q4_K_M.gguf" ]; then
  ~/llama.cpp/build/bin/llama-quantize "$MODELS/Qwen3-8B-bf16.gguf" "$MODELS/Qwen3-8B-Q4_K_M.gguf" Q4_K_M
fi
sha256sum "$MODELS"/Qwen3-8B-bf16.gguf "$MODELS"/Qwen3-8B-Q4_K_M.gguf > "$OUT/checks/gguf_sha256.txt"
cat "$OUT/checks/gguf_sha256.txt"
fi

step "version checks (runbook step 4)"
~/venv/bin/vllm serve --help=all 2>/dev/null | grep -A3 -i "tool-call-parser" > "$OUT/checks/vllm_tool_call_parser.txt" || \
  ~/venv/bin/vllm serve --help 2>&1 | grep -A3 -i "tool-call-parser" > "$OUT/checks/vllm_tool_call_parser.txt" || true
~/venv-sglang/bin/python -m sglang.launch_server --help 2>&1 | grep -A3 -i "tool-call-parser" > "$OUT/checks/sglang_tool_call_parser.txt" || true
~/venv-trtllm/bin/trtllm-serve serve --help 2>&1 | grep -A3 -i "tool_parser" > "$OUT/checks/trtllm_tool_parser.txt" || true
ENGINE_VERSIONS=$(printf '{"sglang": "%s", "tensorrt_llm": "%s", "modelopt": "%s", "trtllm_torch": "%s"}' \
  "$(~/venv-sglang/bin/python -c 'from importlib.metadata import version; print(version("sglang"))')" \
  "$(~/venv-trtllm/bin/python -c 'from importlib.metadata import version; print(version("tensorrt_llm"))')" \
  "$(~/venv-trtllm/bin/python -c 'from importlib.metadata import version; print(version("nvidia-modelopt"))')" \
  "$(~/venv-trtllm/bin/python -c 'import torch; print(torch.__version__)')")
~/venv/bin/python - "$OUT" "$ENGINE_VERSIONS" <<'EOF'
import json, pathlib, sys
from importlib.metadata import version
out = pathlib.Path(sys.argv[1])
revisions = {}
for line in (out / "checks" / "model_paths.txt").read_text().splitlines():
    repo, path = line.split()
    revisions[repo] = pathlib.Path(path).name   # the snapshot directory is the commit hash
versions = {
    "vllm": version("vllm"), "transformers": version("transformers"), "torch": version("torch"),
    "llama_cpp_commit": ((out / "llama_cpp_commit.txt").read_text().strip()
                         if (out / "llama_cpp_commit.txt").exists() else None),
    **json.loads(sys.argv[2]),
    "model_revisions": revisions,
}
(out / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
print(json.dumps(versions, indent=2))
EOF
# Anything other than scripts/gpu/pins.sh would be a different measurement: stop here.
~/venv/bin/python - "$OUT/versions.json" "$VLLM_VERSION" "$TRANSFORMERS_VERSION" "$LLAMA_CPP_COMMIT" \
  "$QWEN3_8B_REVISION" "$QWEN3_8B_AWQ_REVISION" "$SGLANG_VERSION" "$TRTLLM_VERSION" "$MODELOPT_VERSION" \
  "$QWEN3_8B_FP8_REVISION" "$SETUP_LLAMA" <<'EOF'
import json, sys
got = json.load(open(sys.argv[1]))
want = {"vllm": sys.argv[2], "transformers": sys.argv[3], "llama_cpp_commit": sys.argv[4],
        "Qwen/Qwen3-8B": sys.argv[5], "Qwen/Qwen3-8B-AWQ": sys.argv[6], "sglang": sys.argv[7],
        "tensorrt_llm": sys.argv[8], "modelopt": sys.argv[9], "Qwen/Qwen3-8B-FP8": sys.argv[10]}
if sys.argv[11] != "1":
    del want["llama_cpp_commit"]   # not built (SETUP_LLAMA=0)
keys = ("vllm", "transformers", "llama_cpp_commit", "sglang", "tensorrt_llm", "modelopt")
have = {**{k: got.get(k) for k in keys}, **got["model_revisions"]}
wrong = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
if wrong:
    sys.exit("not the pinned versions (installed, pinned): " + json.dumps(wrong))
print("pinned versions: ok")
EOF
grep -c hermes "$OUT/checks/vllm_tool_call_parser.txt" >/dev/null && echo "hermes parser: listed" || echo "hermes parser: NOT FOUND -- check $OUT/checks/vllm_tool_call_parser.txt"
grep -qw qwen "$OUT/checks/sglang_tool_call_parser.txt" && echo "sglang qwen parser: listed" || echo "sglang qwen parser: NOT FOUND -- check $OUT/checks/sglang_tool_call_parser.txt"
grep -qw qwen3 "$OUT/checks/trtllm_tool_parser.txt" && echo "trtllm qwen3 parser: listed" || echo "trtllm qwen3 parser: NOT FOUND -- check $OUT/checks/trtllm_tool_parser.txt"
if [ "$SETUP_LLAMA" = 1 ]; then
step "does llama.cpp honour seed (runbook step 4c)"
~/venv/bin/python "$REPO/scripts/gpu/check_llama_seed.py" || echo "llama seed check failed -- see $OUT/checks/"
fi
step "removing the temporary swap"
sudo swapoff /swapfile-setup
sudo rm -f /swapfile-setup
df -h / ~ > "$OUT/checks/disk.txt"; cat "$OUT/checks/disk.txt"
step "setup done"
touch "$OUT/SETUP_DONE"
