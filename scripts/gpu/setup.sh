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

export PATH="/usr/local/cuda/bin:$PATH"   # the DLAMI installs nvcc there, not always on PATH
echo "== AMI prerequisites"
nvidia-smi | tee "$OUT/checks/nvidia-smi.txt"
nvcc --version | tee "$OUT/checks/nvcc.txt"   # no nvcc -> no CUDA build of llama.cpp; stop here
{ shutdown --show 2>/dev/null || cat /run/systemd/shutdown/scheduled; } 2>&1 | tee "$OUT/checks/shutdown.txt" || true   # --show needs systemd >= 250

echo "== system packages"
sudo apt-get update -q
sudo apt-get install -y -q build-essential cmake git rsync

echo "== uv and the engine environment"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv --allow-existing --python 3.12 ~/venv
# vLLM brings torch and transformers; accelerate is for transformers_hf's device_map.
uv pip install --python ~/venv/bin/python vllm accelerate -r "$REPO/requirements.txt" --torch-backend=auto

echo "== llama.cpp (CUDA, L4 = sm_89)"
if [ ! -d ~/llama.cpp ]; then
  git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
fi
git -C ~/llama.cpp rev-parse HEAD > "$OUT/llama_cpp_commit.txt"   # never recoverable after destroy
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build ~/llama.cpp/build --config Release -j "$(nproc)" --target llama-server

echo "== model weights (Hugging Face cache)"
~/venv/bin/python - <<'EOF' > "$OUT/checks/model_paths.txt"
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-AWQ"):
    print(repo, snapshot_download(repo))
EOF
cat "$OUT/checks/model_paths.txt"

echo "== BF16 GGUF from the same safetensors (docs/engine-performance-plan.md)"
# A separate environment: the converter's requirements pin a CPU torch that
# would replace vLLM's CUDA build.
uv venv --allow-existing --python 3.12 ~/venv-convert
uv pip install --python ~/venv-convert/bin/python -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
SNAPSHOT=$(awk '$1 == "Qwen/Qwen3-8B" {print $2}' "$OUT/checks/model_paths.txt")
if [ ! -f "$MODELS/Qwen3-8B-bf16.gguf" ]; then
  ~/venv-convert/bin/python ~/llama.cpp/convert_hf_to_gguf.py "$SNAPSHOT" \
    --outtype bf16 --outfile "$MODELS/Qwen3-8B-bf16.gguf"
fi

echo "== version checks (runbook step 4)"
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
echo "== does llama.cpp honour seed (runbook step 4c)"
~/venv/bin/python "$REPO/scripts/gpu/check_llama_seed.py" || echo "llama seed check failed -- see $OUT/checks/"
touch "$OUT/SETUP_DONE"
echo "setup done"
