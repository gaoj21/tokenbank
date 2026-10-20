#!/bin/bash
# gemma-4-31B-it QAT Q4_0 GGUF via self-built CUDA llama-server (sm_86).
# Single GPU; OpenAI-compatible endpoint on :8001.
#
# --reasoning off is required for protocol parity with the Qwen runs, which
# disable thinking through chat_template_kwargs {enable_thinking: False}. Newer
# llama.cpp builds turn thinking on by default and route it to
# `message.reasoning_content`; the harness reads only `message.content`, so with
# thinking on and MAX_TOKENS=1024 the model spends the whole cap thinking and
# returns an empty answer (finish_reason=length, content=""). Every task would
# score zero for a reason that has nothing to do with the budget policy.
GGUF=/home/j0018/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-q4_0-gguf/snapshots/59dde24573e7e61570dba08b18a2e1fe246955ed/gemma-4-31B_q4_0-it.gguf
export LD_LIBRARY_PATH="/home/j0018/anaconda3/envs/vllm/cuda-compat:/home/j0018/anaconda3/envs/vllm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
exec /home/j0018/Desktop/tokenbank/tools/llama.cpp/build/bin/llama-server \
  -m "$GGUF" \
  --alias gemma-4-31B-it \
  --port "${PORT:-8001}" \
  -ngl 99 \
  --tensor-split 1,1,1 \
  -c 131072 \
  -np 16 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --reasoning off
