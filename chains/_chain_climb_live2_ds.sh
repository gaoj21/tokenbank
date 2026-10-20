#!/bin/bash
# DeepSeek live CLIMB with the final (combined-test) thresholds, for MATH and MuSiQue.
cd /home/j0018/Desktop/tokenbank
until curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:18000/v1/models | grep -q 200; do sleep 30; done
echo "[$(date '+%F %T')] server ready"
run() { local T=$1 t=$2 url=$3 model=$4 w=$5 th=$6 ids=$7; shift 7
  [ -s traces/tasks_sweep_$T$t.jsonl ] && { echo SKIP $t; return; }
  echo "[$(date '+%F %T')] START $T$t theta=$th"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=$w \
    MAX_TOKENS=4096 THETA_LIVE=$th TASK_IDS_FILE=$ids OUT_TAG=$t CONFIG_ONLY=CLIMB-Live "$@" \
    .venv/bin/python -u _budget_sweep_v2.py > logs/${T}$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $T$t"; }
D=DeepSeek-V4.1-Flash; U=http://127.0.0.1:18000/v1; I=results/climb_live
run $D _climb2_ma24 $U deepseek-v4.1-flash 40 -0.245 $I/test_ids_DeepSeek_MATH.json BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test L01_THINKING_OFF=1 &
run $D _climb2_mu24 $U deepseek-v4.1-flash 40 -0.275 $I/test_ids_DeepSeek_MuSiQue.json BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test L01_THINKING_OFF=1 &
wait; echo "[$(date '+%F %T')] CLIMB LIVE2 DS DONE"
