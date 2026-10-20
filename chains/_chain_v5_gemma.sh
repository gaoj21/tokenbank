#!/bin/bash
# v5 final confirmation on Gemma-4-31B-it (llama-server :8001), single seed.
cd /home/j0018/Desktop/tokenbank
run() {
  local bench=$1 tag=$2 log=$3; shift 3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=gemma-4-31B-it \
  BENCHMARK=$bench BUDGETS=24000 SEEDS=42 MAX_WORKERS=16 SPLIT=test \
  V5=1 CONFIG_ONLY="V5-A0-Full,V5-A4-NoM4,V5-Unconstrained" \
  OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run GAIA _v5gem_ga _v5gem_gaia.log GAIA_N=1000
run MATH _v5gem_ma _v5gem_math.log MATH_N=500
