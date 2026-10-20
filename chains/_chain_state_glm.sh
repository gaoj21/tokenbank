#!/bin/bash
# State-policy comparison on GLM, cap 4096, budgets 24k and 8k. The rebuild arm (V5-Unc-Fmt)
# already exists at both budgets at this cap (glmx4k_*24, glmr4k_*8). MuSiQue is subsampled to
# 600 items because six arms over 1,934 items would dominate the run; pairing is by task_id.
cd /home/j0018/Desktop/tokenbank
while ! grep -q "GLM SEG DONE" logs/_chain_glm_seg.log 2>/dev/null; do sleep 60; done
T=GLM-5.3-Flash; ARMS=ST-Accum,ST-Window2,ST-Summary,ST-Slots,ST-Slots-M1,ST-Rebuild-M1
run() { local t=$1 url=$2 model=$3; shift 3
  [ -s "traces/tasks_sweep_$T$t.jsonl" ] && { echo "SKIP $t"; return; }
  echo "[$(date '+%F %T')] START $t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 \
      MAX_WORKERS=32 MAX_TOKENS=4096 SEGMENT_ACCOUNTING=1 OUT_TAG=$t CONFIG_ONLY=$ARMS "$@" \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
E1="https://llm3.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8"; E2="https://llm5.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8"
E3="https://llm52.jierungogogo.com/v1 zai-org/GLM-5.3-Flash"; E4="https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash"
E5="https://llm57.jierungogogo.com/v1 zai-org/GLM-5.3-Flash"
{ run _glmst_ma24 $E1 BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test; run _glmst_ma8 $E1 BENCHMARK=MATH MATH_N=500 BUDGETS=8000 SPLIT=test; } &
{ run _glmst_ga24 $E2 BENCHMARK=GAIA GAIA_N=1000 BUDGETS=24000 SPLIT=test; run _glmst_ga8 $E2 BENCHMARK=GAIA GAIA_N=1000 BUDGETS=8000 SPLIT=test; } &
{ run _glmst_fr24 $E3 BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test; run _glmst_fr8 $E3 BENCHMARK=FRAMES BUDGETS=8000 SPLIT=test; } &
run _glmst_mu24 $E4 BENCHMARK=MUSIQUE MUSIQUE_N=600 BUDGETS=24000 SPLIT=test &
run _glmst_mu8  $E5 BENCHMARK=MUSIQUE MUSIQUE_N=600 BUDGETS=8000 SPLIT=test &
wait; echo "[$(date '+%F %T')] GLM STATE DONE"
