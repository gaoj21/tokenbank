#!/bin/bash
# Live CLIMB with the final (combined-test) thresholds, for settings whose threshold changed.
# GLM only for now; DeepSeek MATH / MuSiQue wait for the remote GPUs.
cd /home/j0018/Desktop/tokenbank
run() { local T=$1 t=$2 url=$3 model=$4 w=$5 th=$6 ids=$7; shift 7
  [ -s traces/tasks_sweep_$T$t.jsonl ] && { echo SKIP $t; return; }
  echo "[$(date '+%F %T')] START $T$t theta=$th"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=$w \
    MAX_TOKENS=4096 THETA_LIVE=$th TASK_IDS_FILE=$ids OUT_TAG=$t CONFIG_ONLY=CLIMB-Live "$@" \
    .venv/bin/python -u _budget_sweep_v2.py > logs/${T}$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $T$t"; }
G=GLM-5.3-Flash; I=results/climb_live
run $G _climb2_fr24 https://llm52.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 -0.29 $I/test_ids_GLM_FRAMES.json BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test &
run $G _climb2_mu24 https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 -0.5 $I/test_ids_GLM_MuSiQue.json BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test &
wait; echo "[$(date '+%F %T')] CLIMB LIVE2 DONE"
