#!/bin/bash
# Full experiment chain: Mistral(reasoning) -> Laguna(fast) -> Gemma informed-static ablation.
# Single wrapper, no stacked waiters: everything sequential in one process.
cd /root/tokenbank
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN} HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=30 SEEDS=42 MAX_WORKERS=32

switch_model() {
  local M="$1"
  for p in $(pgrep -f "[b]in/vllm"; pgrep -f "[V]LLM::EngineCore"); do kill -9 "$p" 2>/dev/null; done
  sleep 15
  VLLM_MODEL="$M" nohup bash chains/_start_vllm.sh > "_vllm_boot_$(basename "$M").log" 2>&1 &
  for i in $(seq 1 100); do
    sleep 30
    curl -s --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -qi "$(basename "$M")" && return 0
    pgrep -f "[b]in/vllm" >/dev/null || return 1   # server process died
  done
  return 1
}

run_sweeps() {  # $1=tag for logs
  local TAG="$1"
  for spec in "GAIA GAIA_N=200" "FRAMES FRAMES_N=824" "MUSIQUE MUSIQUE_N=2417" "MATH MATH_N=3669"; do
    set -- $spec
    echo "=== sweep $1 @ $VLLM_MODEL ($TAG) ==="
    env BENCHMARK=$1 $2 BUDGETS=$BUDGET_LIST OUT_TAG=$OUT_TAG .venv/bin/python -u _budget_sweep_vllm.py > "_sweep_${TAG}_${1,,}.log" 2>&1
  done
}

echo "=== PHASE 1: Mistral reasoning (high, max_tokens=4096) ==="
export VLLM_MODEL=mistralai/Mistral-Small-4-119B-2603-NVFP4 REASONING_EFFORT=high MAX_TOKENS=4096
export BUDGET_LIST=8000,16000,32000,64000,128000 OUT_TAG=_reason
switch_model "$VLLM_MODEL" || { echo "MISTRAL SERVER FAILED"; exit 1; }
run_sweeps mistral_reason

echo "=== PHASE 2: Laguna fast mode ==="
unset REASONING_EFFORT
export VLLM_MODEL=poolside/Laguna-S-2.1-NVFP4 MAX_TOKENS=1024
export BUDGET_LIST=2000,4000,8000,16000,32000,64000 OUT_TAG=
switch_model "$VLLM_MODEL" || { echo "LAGUNA SERVER FAILED"; exit 1; }
run_sweeps laguna

echo "=== PHASE 3: Gemma informed-static ablation ==="
export VLLM_MODEL=google/gemma-4-31B-it
switch_model "$VLLM_MODEL" || { echo "GEMMA SERVER FAILED"; exit 1; }
for spec in "GAIA GAIA_N=200" "FRAMES FRAMES_N=824"; do
  set -- $spec
  echo "=== informed-static $1 @ gemma ==="
  env BENCHMARK=$1 $2 BUDGETS=4000,8000,16000,32000,64000 CALIB_N=20 .venv/bin/python -u _informed_static_baseline.py > "_informed_${1,,}_gemma.log" 2>&1
done

echo CHAIN_ALL_DONE
