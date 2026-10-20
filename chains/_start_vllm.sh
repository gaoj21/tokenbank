#!/bin/bash
# Start the local vLLM server for Token Bank experiments.
# OpenAI-compatible endpoint at http://127.0.0.1:8000/v1
# Usage:  bash chains/_start_vllm.sh                      (default: Qwen/Qwen3.6-35B-A3B)
#         VLLM_MODEL=google/gemma-4-31B bash chains/_start_vllm.sh
cd "$(dirname "$0")"
MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-35B-A3B}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/${MODEL##*/}}"
PORT="${PORT:-8000}"

# FlashInfer JIT (sm_120 fused MoE) requires nvcc >= 12.9; the system CUDA is
# 12.8, so point CUDA_HOME at the pip-installed CUDA 13.3 toolkit.
export CUDA_HOME="$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$(pwd)/.venv/bin:$CUDA_HOME/bin:$PATH"
# Link/runtime search paths for the pip CUDA 13 toolkit (JIT .so links -lcudart/-lnvrtc)
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# pip toolchain mixes nvcc 13.3 with CUDA 13.0 headers; bypass CCCL's
# compiler/headers equality check (official escape hatch).
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

# Model-specific serve flags
EXTRA_ARGS=""
case "$MODEL" in
  *istral*)
    # Mistral Small 4 (model card): MLA attention + mistral tool/reasoning parsers
    EXTRA_ARGS="--attention-backend TRITON_MLA --tool-call-parser mistral --reasoning-parser mistral"
    ;;
  *aguna*)
    # Laguna S 2.1 (model card): poolside parsers; NVFP4 weights are ~100GB
    # (mixed precision), offload part of them to host RAM to fit 98GB GPU
    EXTRA_ARGS="--tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --cpu-offload-gb ${CPU_OFFLOAD_GB:-25}"
    ;;
esac

exec .venv/bin/vllm serve "$MODEL_DIR" \
  --served-model-name "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  $EXTRA_ARGS \
  2>&1 | tee -a "_vllm_server_${MODEL##*/}.log"
