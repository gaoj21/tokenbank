#!/bin/bash
# v5.2 Phase Ab: v5.1 own ablations — GAIA + MATH @24k (Ab3 first, then Ab1, Ab2)
cd /home/j0018/Desktop/tokenbank
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done
run() {
  local bench=$1 tag=$2 log=$3; shift 3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=test \
  V5=1 CONFIG_ONLY="V5.1-Ab3-NoM2,V5.1-Ab1-NoM1,V5.1-Ab2-NoNotes" \
  OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run GAIA _v52ab_ga _v52ab_gaia.log GAIA_N=1000
run MATH _v52ab_ma _v52ab_math.log MATH_N=500
