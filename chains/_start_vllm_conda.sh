#!/bin/bash
# Local vLLM server via the conda `vllm` env (vllm 0.27.1, torch 2.13+cu130).
# Driver 535 is too old for CUDA 13, so LD_LIBRARY_PATH points at the
# cuda-compat userspace driver installed in the env (libcuda.so.610).
# Model: Qwen/Qwen3.6-27B-FP8 (~27GB) — needs TP=2 on 24GB A5000s.
# OpenAI-compatible endpoint at http://127.0.0.1:8001/v1 (8000 is taken).
CONDA_ENV=/home/j0018/anaconda3/envs/vllm
MODEL_DIR=$(ls -d /home/j0018/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/*/ | head -1)
export LD_LIBRARY_PATH="$CONDA_ENV/cuda-compat:$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CONDA_ENV/bin:$PATH"
export CUDA_HOME="$CONDA_ENV"
export LIBRARY_PATH="$CONDA_ENV/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export VLLM_LOGGING_LEVEL=INFO
exec "$CONDA_ENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name Qwen/Qwen3.6-27B-FP8 \
  --port "${PORT:-8001}" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  --max-num-seqs 128
