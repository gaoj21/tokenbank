#!/bin/bash
# Local vLLM server via the conda `vllm` env, Gemma-4-31B-IT QAT Q4_0 GGUF.
# GGUF in vLLM is single-GPU only (no TP); q4_0 31B ~18.5GB fits one 24GB
# A5000 with a few GB left for KV, so keep max-model-len/num-seqs modest.
# Tokenizer comes from the unsloth mirror (google/gemma-4-31B-it is gated).
# OpenAI-compatible endpoint at http://127.0.0.1:8001/v1.
CONDA_ENV=/home/j0018/anaconda3/envs/vllm
GGUF=$(ls /home/j0018/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-q4_0-gguf/snapshots/*/gemma-4-31B_q4_0-it.gguf | head -1)
export LD_LIBRARY_PATH="$CONDA_ENV/cuda-compat:$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CONDA_ENV/bin:$PATH"
export CUDA_HOME="$CONDA_ENV"
export LIBRARY_PATH="$CONDA_ENV/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_LOGGING_LEVEL=INFO
exec "$CONDA_ENV/bin/vllm" serve "$GGUF" \
  --served-model-name gemma-4-31B-it \
  --tokenizer "${GEMMA_TOKENIZER:-unsloth/gemma-4-31B-it}" \
  --port "${PORT:-8001}" \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  --max-num-seqs 64
