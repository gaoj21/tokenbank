#!/bin/bash
# Same for DeepSeek; starts after the DeepSeek live reruns finish.
cd /home/j0018/Desktop/tokenbank
until grep -q "CLIMB LIVE2 DS DONE" logs/_chain_climb_live2_ds.log 2>/dev/null; do sleep 60; done
run() { local T=$1 t=$2 url=$3 model=$4 w=$5; shift 5
  [ -s traces/tasks_sweep_$T$t.jsonl ] && { echo SKIP $t; return; }
  echo "[$(date '+%F %T')] START $T$t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=$w \
    MAX_TOKENS=4096 THETA_LIVE=99 OUT_TAG=$t CONFIG_ONLY=CLIMB-Live "$@" \
    .venv/bin/python -u _budget_sweep_v2.py > logs/${T}$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $T$t"; }
D=DeepSeek-V4.1-Flash; U=http://127.0.0.1:18000/v1
run $D _climbx_ma24 $U deepseek-v4.1-flash 30 BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test L01_THINKING_OFF=1 &
run $D _climbx_ga24 $U deepseek-v4.1-flash 20 BENCHMARK=GAIA GAIA_N=1000 BUDGETS=24000 SPLIT=test L01_THINKING_OFF=1 &
run $D _climbx_fr24 $U deepseek-v4.1-flash 30 BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test L01_THINKING_OFF=1 &
run $D _climbx_mu24 $U deepseek-v4.1-flash 40 BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test L01_THINKING_OFF=1 &
wait; echo "[$(date '+%F %T')] CLIMBX DS DONE"
