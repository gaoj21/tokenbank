#!/bin/bash
# DeepSeek MuSiQue shards 1 and 2 (started early), then merge all shards for both models.
cd /home/j0018/Desktop/tokenbank
run() { local T=$1 t=$2 url=$3 model=$4 w=$5 ids=$6; shift 6
  [ -s traces/tasks_sweep_$T$t.jsonl ] && { echo SKIP $t; return; }
  echo "[$(date '+%F %T')] START $T$t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=$w \
    MAX_TOKENS=4096 THETA_LIVE=99 OUT_TAG=$t CONFIG_ONLY=CLIMB-Live TASK_IDS_FILE=$ids "$@" \
    .venv/bin/python -u _budget_sweep_v2.py > logs/${T}$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $T$t"; }
D=DeepSeek-V4.1-Flash; U=http://127.0.0.1:18000/v1; S=results/climb_live
M="BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test"
run $D _climbx_mu24s1 $U deepseek-v4.1-flash 40 $S/mu_shard1.json $M L01_THINKING_OFF=1 &
run $D _climbx_mu24s2 $U deepseek-v4.1-flash 40 $S/mu_shard2.json $M L01_THINKING_OFF=1 &
wait
until [ -s traces/tasks_sweep_${D}_climbx_mu24s0.jsonl ] && [ -s traces/tasks_sweep_GLM-5.3-Flash_climbx_mu24s0.jsonl ] \
   && [ -s traces/tasks_sweep_GLM-5.3-Flash_climbx_mu24s1.jsonl ] && [ -s traces/tasks_sweep_GLM-5.3-Flash_climbx_mu24s2.jsonl ]; do sleep 30; done
for T in GLM-5.3-Flash $D; do cat traces/tasks_sweep_${T}_climbx_mu24s{0,1,2}.jsonl > traces/tasks_sweep_${T}_climbx_mu24.jsonl; done
echo "[$(date '+%F %T')] CLIMBX MU2 DONE"
