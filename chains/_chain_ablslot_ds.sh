#!/bin/bash
# Single-component ablations of the slot state (plan line only, capped JSON critique only, tool
# note only), Accept stop rule, cap 4096, 24k. Identifies which slot discards what revision needs.
cd /home/j0018/Desktop/tokenbank
while ! grep -q "DS STATE DONE" logs/_chain_state_ds.log 2>/dev/null; do sleep 60; done
T=DeepSeek-V4.1-Flash; ARMS=ST-PlanOnly,ST-CritOnly,ST-ToolOnly,ST-CritOnly-R1k,ST-Slots-R1k
run() { local t=$1 url=$2 model=$3 w=$4; shift 4
  [ -s "traces/tasks_sweep_$T$t.jsonl" ] && { echo "SKIP $t"; return; }
  echo "[$(date '+%F %T')] START $t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 \
      MAX_WORKERS=$w MAX_TOKENS=4096 SEGMENT_ACCOUNTING=1 OUT_TAG=$t CONFIG_ONLY=$ARMS "$@" \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }

for b in "MATH MATH_N=500:ma" "GAIA GAIA_N=1000:ga" "FRAMES x=1:fr" "MUSIQUE MUSIQUE_N=600:mu"; do
  set -- ${b%%:*}; t=${b##*:}
  run _dsab_${t}24 http://127.0.0.1:18000/v1 deepseek-v4.1-flash 128 BENCHMARK=$1 $2 BUDGETS=24000 SPLIT=test
done
wait; echo "[$(date '+%F %T')] ABL DONE"
