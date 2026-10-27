# The versions every GPU apply runs (sourced by setup.sh). Fixed to the set the
# 2026-10-18 smoke ran end to end (runs/gpu-20261018/versions.json), so the
# pilot and the production runs measure the same engines. Unpinned, the second
# apply (2026-10-21) got llama.cpp b11785, vLLM 0.32.0 and transformers 5.19.0.
# setup.sh stops if what it installed differs from these.
LLAMA_CPP_COMMIT=848bbda9041decd87904f7eb5a2990863ecce03b
VLLM_VERSION=0.31.0
TRANSFORMERS_VERSION=5.17.0
# Not pinned at download (vLLM and transformers load the models by id from the
# cache's main ref); setup.sh only stops if the downloaded revision is another one.
QWEN3_8B_REVISION=b968826d9c46dd6066d109eabc6255188de91218
QWEN3_8B_AWQ_REVISION=4da05a8edb55c6046cce958586c33b61da07bb79
# The continuous-batching engines of docs/adr/0009 (each in its own environment),
# the newest stable releases at least two weeks old on 2026-10-27:
# SGLang 0.5.21 (2026-10-02; 0.5.22 was 8 days old) and TensorRT-LLM 1.2.1
# (2026-04-20; everything newer is a release candidate). ModelOpt is the
# version TensorRT-LLM 1.2.1 requires (nvidia-modelopt[torch]~=0.37.0).
SGLANG_VERSION=0.5.21
TRTLLM_VERSION=1.2.1
TRTLLM_TORCH_VERSION=2.9.1
MODELOPT_VERSION=0.37.0
QWEN3_8B_FP8_REVISION=220b46e3b2180893580a4454f21f22d3ebb187d3
