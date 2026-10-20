#!/bin/bash
# All-escalate records with the deployed CLIMB configuration (THETA_LIVE=99: every task climbs
# L0 -> L1 -> seeded L2), full test split, GLM. Gives replay data identical to deployment.
cd /home/j0018/Desktop/tokenbank
run() { local T=$1 t=$2 url=$3 model=$4 w=$5; shift 5
  [ -s traces/tasks_sweep_$T$t.jsonl ] && { echo SKIP $t; return; }
  echo "[$(date '+%F %T')] START $T$t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=$w \
    MAX_TOKENS=4096 THETA_LIVE=99 OUT_TAG=$t CONFIG_ONLY=CLIMB-Live "$@" \
    .venv/bin/python -u _budget_sweep_v2.py > logs/${T}$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $T$t"; }
G=GLM-5.3-Flash
run $G _climbx_ma24 https://llm3.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test &
run $G _climbx_ga24 https://llm5.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 BENCHMARK=GAIA GAIA_N=1000 BUDGETS=24000 SPLIT=test &
run $G _climbx_fr24 https://llm52.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test &
run $G _climbx_mu24 https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test &
wait; echo "[$(date '+%F %T')] CLIMBX GLM DONE"
