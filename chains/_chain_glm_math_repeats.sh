#!/bin/bash
# Run-to-run variation on GLM MATH (FP8, live test half): two more live CLIMB runs at the certified
# threshold and one more plain-pipeline run. All results are reported, none is selected.
cd /home/j0018/Desktop/tokenbank
G=GLM-5.3-Flash; I=results/climb_live; X="BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test"
TH=$(.venv/bin/python -c "import json; print(json.load(open('results/climb_live_plan.json'))['GLM|MATH']['threshold_v3'])")
echo "threshold=$TH"
run() { local t=$1 h=$2 cfg=$3 extra=$4
  echo "[$(date '+%F %T')] START $t ($h)"
  env OPENAI_BASE_URL=https://$h.ruoyu-wang.com/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=GLM-5.3-Flash-FP8 MODEL_TAG=$G SEEDS=42 TEMPERATURE=0 V5=1 \
    MAX_WORKERS=32 MAX_TOKENS=4096 OUT_TAG=$t CONFIG_ONLY=$cfg TASK_IDS_FILE=$I/test_ids_GLM_MATH.json $extra $X .venv/bin/python -u _budget_sweep_v2.py > logs/$G$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t APIERR=$(grep -c APIERR logs/$G$t.log)"; }
run _climb3r1_ma24 llm3 CLIMB-Live "THETA_LIVE=$TH" &
run _climb3r2_ma24 llm4 CLIMB-Live "THETA_LIVE=$TH" &
run _pipe3r1_ma24 llm5 V5-Unc-Fmt "" &
wait; echo "[$(date '+%F %T')] REPEATS DONE"
