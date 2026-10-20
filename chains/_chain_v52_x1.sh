#!/bin/bash
# v5.2 X1: Gemma-4-31B-it × V5.1-NoPool @24k (MATH+GAIA) — cross-model check of the FIX.
# Requires llama-server (gemma) on :8001 — stop vLLM first.
cd /home/j0018/Desktop/tokenbank
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done
run() {
  local bench=$1 tag=$2 log=$3; shift 3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=gemma-4-31B-it \
  BENCHMARK=$bench BUDGETS=24000 SEEDS=42 MAX_WORKERS=8 SPLIT=test \
  V5=1 CONFIG_ONLY="V5.1-NoPool" \
  OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run MATH _v52x1_ma _v52x1_math.log MATH_N=500
run GAIA _v52x1_ga _v52x1_gaia.log GAIA_N=1000
