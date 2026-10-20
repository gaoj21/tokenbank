#!/bin/bash
# Local vLLM server for Token Bank experiments (this machine: 3x RTX A5000 24GB,
# driver 535). Model: community AWQ-4bit quant of Qwen3.6-35B-A3B (~20GB) —
# official FP8 needs sm_89+, A5000 is sm_86. Tensor-parallel over 2 GPUs for
# KV headroom. OpenAI-compatible endpoint at http://127.0.0.1:8000/v1
cd "$(dirname "$0")"
MODEL_DIR=/home/j0018/.cache/huggingface/hub/models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit/snapshots/00fcea2d3bcf5389b518d4fc082e5590e0ba4844
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_LOGGING_LEVEL=INFO
exec .venv/bin/vllm serve "$MODEL_DIR" \
  --served-model-name Qwen/Qwen3.6-35B-A3B \
  --port "${PORT:-8000}" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 16
