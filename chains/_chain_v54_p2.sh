#!/bin/bash
# v5.3 Z1b follow-up, P2: Gemma-4-31B-it x V5-Unc-Fmt x MATH 24k.
# The missing format-aligned baseline for the cross-model arm. X1 reported Gemma
# +25.8pp, but against Unc (no format) — the same comparison E1 showed to be 57%
# format on Qwen. If Gemma's Unc+Fmt lands near NoPool's 64.8%, the cross-model SR
# claim collapses the way MuSiQue's did; if it stays low, the claim survives.
#
# Needs llama-server on :8001 across all 3 GPUs -> vLLM must go first.
set -e
cd /home/j0018/Desktop/tokenbank

# 1. wait for any sweep still running
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done

# 2. stop vLLM and wait for the GPUs to actually drain
pkill -f "vllm serve" || true
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  [ "$used" -lt 2000 ] && break
  sleep 10
done
echo "GPU drained: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

# 3. start llama-server (gemma) and wait for it to answer
nohup bash chains/_start_gemma_llama.sh > logs/_gemma_server_v54.log 2>&1 &
for i in $(seq 1 120); do
  curl -s -m 5 http://127.0.0.1:8001/v1/models | grep -q gemma && break
  sleep 10
done
curl -s -m 5 http://127.0.0.1:8001/v1/models | head -c 200; echo

# 4. run (MAX_WORKERS=8: llama.cpp -np 16, same as the X1 run this compares against)
env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=gemma-4-31B-it \
  BENCHMARK=MATH BUDGETS=24000 SEEDS=42 MAX_WORKERS=8 SPLIT=test \
  V5=1 CONFIG_ONLY="V5-Unc-Fmt" OUT_TAG=_v54b_ma_fmt TEMPERATURE=0 MATH_N=500 \
  .venv/bin/python -u _budget_sweep_v2.py > logs/_v54b_math_gemma_fmt.log 2>&1
