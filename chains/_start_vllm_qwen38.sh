#!/bin/bash
# Local vLLM server, Qwen3.8-27B-FP8, on this machine (3x RTX A5000 24GB, sm_86,
# driver 535 / CUDA 12.2).
#
# Replaces the Qwen3.6-27B-FP8 server that previously ran on :8001.
#
# Notes on why the environment below is what it is:
#   - The anaconda `vllm` env ships torch 2.13+cu130, which refuses to init CUDA
#     against this machine's 12020 driver. `envs/vllm/cuda-compat` holds forward-
#     compatible driver libs (libcuda.so.610.43.02); putting it FIRST on
#     LD_LIBRARY_PATH is what makes the whole thing work. Drop it and the server
#     dies at startup with "NVIDIA driver on your system is too old".
#   - A5000 is sm_86 and native FP8 needs sm_89+. vLLM runs these weights through
#     the Marlin FP8 path instead: weights stay FP8 in memory (~31GB), and are
#     dequantized inside the kernel. Slower than native FP8, but it fits.
#   - Tool calls need BOTH --enable-auto-tool-choice and --tool-call-parser, or
#     vLLM rejects any request carrying `tools` outright. The parser must match
#     the chat template: this model emits the XML form
#     (<tool_call><function=name><parameter=p>v</parameter></function></tool_call>),
#     not Hermes JSON, so `hermes` would parse nothing. qwen3_xml and qwen3_coder
#     are the same class; either name works.
#   - Without --reasoning-parser the chain of thought lands inside `content`
#     together with a stray </think>, which the agent then feeds back as context.
#   - Qwen3.8-27B is a hybrid-attention model: 64 layers, only 16 full-attention
#     and 48 linear-attention. KV per token is therefore ~4x smaller than a dense
#     model of the same size, which is why MAX_MODEL_LEN can be far above the
#     8192 the 3.6 server was pinned to.
#
# Usage:  bash chains/_start_vllm_qwen38.sh
#         MAX_MODEL_LEN=65536 MAX_NUM_SEQS=8 bash chains/_start_vllm_qwen38.sh

set -euo pipefail
cd "$(dirname "$0")"

VLLM_ENV=/home/j0018/anaconda3/envs/vllm
MODEL_DIR="${MODEL_DIR:-$(ls -d /home/j0018/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)}"

export CUDA_HOME="$VLLM_ENV"
export LD_LIBRARY_PATH="$VLLM_ENV/cuda-compat:$VLLM_ENV/lib"
export LIBRARY_PATH="$VLLM_ENV/lib"
export PATH="$VLLM_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_LOGGING_LEVEL=INFO

PORT="${PORT:-8001}"
TP="${TP:-2}"
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

exec "$VLLM_ENV/bin/python3.10" "$VLLM_ENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name Qwen/Qwen3.8-27B-FP8 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-qwen3_xml}" \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  2>&1 | tee -a "_vllm_server_Qwen3.8-27B-FP8.log"
