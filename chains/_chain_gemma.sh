#!/bin/bash
cd /home/j0018/Desktop/tokenbank
run() {  # bench budgets configs tag log [extra_env...]
  local bench=$1 budgets=$2 configs=$3 tag=$4 log=$5; shift 5
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=gemma-4-31B-it \
  BENCHMARK=$bench BUDGETS=$budgets SEEDS=42 MAX_WORKERS=16 CONFIG_ONLY="$configs" \
  OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
# GAIA full budget curve first (152 tasks, cheapest, measures real speed)
run GAIA 16000 "Unconstrained,Static Quota,Improved+Compact" _ga16k_gem _gemma_gaia16k.log
run GAIA 24000 "Static Quota,Improved+Compact"               _ga24k_gem _gemma_gaia24k.log
run GAIA 32000 "Static Quota,Improved+Compact"               _ga32k_gem _gemma_gaia32k.log
