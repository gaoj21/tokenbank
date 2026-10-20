#!/bin/bash
# State-policy comparison on DeepSeek (remote H20, one server), cap 4096, budgets 24k and 8k. The rebuild arm (V5-Unc-Fmt)
# already exists at both budgets at this cap (dsx4k_*24, dsr4k_*8). MuSiQue is subsampled to
# 600 items because six arms over 1,934 items would dominate the run; pairing is by task_id.
cd /home/j0018/Desktop/tokenbank
while ! grep -q "DS FINAL ALL DONE" logs/_chain_ds_final.log 2>/dev/null; do sleep 60; done
T=DeepSeek-V4.1-Flash; ARMS=ST-Accum,ST-Window2,ST-Summary,ST-Slots,ST-Slots-M1,ST-Rebuild-M1
run() { local t=$1 url=$2 model=$3; shift 3
  [ -s "traces/tasks_sweep_$T$t.jsonl" ] && { echo "SKIP $t"; return; }
  echo "[$(date '+%F %T')] START $t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 \
      MAX_WORKERS=128 MAX_TOKENS=4096 SEGMENT_ACCOUNTING=1 OUT_TAG=$t CONFIG_ONLY=$ARMS "$@" \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
E="http://127.0.0.1:18000/v1 deepseek-v4.1-flash"
{ run _dsst_ma24 $E BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test; run _dsst_ma8 $E BENCHMARK=MATH MATH_N=500 BUDGETS=8000 SPLIT=test; } &
{ run _dsst_ga24 $E BENCHMARK=GAIA GAIA_N=1000 BUDGETS=24000 SPLIT=test; run _dsst_ga8 $E BENCHMARK=GAIA GAIA_N=1000 BUDGETS=8000 SPLIT=test; } &
{ run _dsst_fr24 $E BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test; run _dsst_fr8 $E BENCHMARK=FRAMES BUDGETS=8000 SPLIT=test; } &
run _dsst_mu24 $E BENCHMARK=MUSIQUE MUSIQUE_N=600 BUDGETS=24000 SPLIT=test &
run _dsst_mu8  $E BENCHMARK=MUSIQUE MUSIQUE_N=600 BUDGETS=8000 SPLIT=test &
wait; echo "[$(date '+%F %T')] DS STATE DONE"
