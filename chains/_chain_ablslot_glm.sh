#!/bin/bash
# Single-component ablations of the slot state (plan line only, capped JSON critique only, tool
# note only), Accept stop rule, cap 4096, 24k. Identifies which slot discards what revision needs.
cd /home/j0018/Desktop/tokenbank
while ! grep -q "GLM STATE DONE" logs/_chain_state_glm.log 2>/dev/null; do sleep 60; done
T=GLM-5.3-Flash; ARMS=ST-PlanOnly,ST-CritOnly,ST-ToolOnly
run() { local t=$1 url=$2 model=$3 w=$4; shift 4
  [ -s "traces/tasks_sweep_$T$t.jsonl" ] && { echo "SKIP $t"; return; }
  echo "[$(date '+%F %T')] START $t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 \
      MAX_WORKERS=$w MAX_TOKENS=4096 SEGMENT_ACCOUNTING=1 OUT_TAG=$t CONFIG_ONLY=$ARMS "$@" \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }

run _glmab_ma24 https://llm3.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test &
run _glmab_ga24 https://llm5.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 BENCHMARK=GAIA GAIA_N=1000 BUDGETS=24000 SPLIT=test &
run _glmab_fr24 https://llm52.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test &
run _glmab_mu24 https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 BENCHMARK=MUSIQUE MUSIQUE_N=600 BUDGETS=24000 SPLIT=test &
wait; echo "[$(date '+%F %T')] ABL DONE"
