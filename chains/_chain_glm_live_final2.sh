#!/bin/bash
# GLM FRAMES / MuSiQue live runs on FP8 with the thresholds certified by _chain_glm_live_final.sh
# (results/climb_live_plan.json: FRAMES -0.12, MuSiQue -0.415).
cd /home/j0018/Desktop/tokenbank
G=GLM-5.3-Flash; I=results/climb_live
M="BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test"; F="BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test"
TF=$(.venv/bin/python -c "import json; print(json.load(open('$I/../climb_live_plan.json'))['GLM|FRAMES']['threshold_v3'])")
TM=$(.venv/bin/python -c "import json; print(json.load(open('$I/../climb_live_plan.json'))['GLM|MuSiQue']['threshold_v3'])")
echo "[$(date '+%F %T')] thresholds FRAMES=$TF MuSiQue=$TM"
[ -n "$TF" ] && [ -n "$TM" ] || { echo "empty threshold, abort"; exit 1; }
run() { local t=$1 h=$2 th=$3 ids=$4; shift 4
  echo "[$(date '+%F %T')] START $t ($h) theta=$th"
  env OPENAI_BASE_URL=https://$h.ruoyu-wang.com/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=GLM-5.3-Flash-FP8 MODEL_TAG=$G SEEDS=42 TEMPERATURE=0 V5=1 \
    MAX_WORKERS=32 MAX_TOKENS=4096 OUT_TAG=$t CONFIG_ONLY=CLIMB-Live THETA_LIVE=$th TASK_IDS_FILE=$ids "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$G$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t APIERR=$(grep -c APIERR logs/$G$t.log)"; }
run _climb3_fr24 llm3 $TF $I/test_ids_GLM_FRAMES.json $F &
run _climb3_mu24x0 llm4 $TM $I/live_mu_x0.json $M &
run _climb3_mu24x1 llm5 $TM $I/live_mu_x1.json $M &
wait
cat traces/tasks_sweep_${G}_climb3_mu24x{0,1}.jsonl > traces/tasks_sweep_${G}_climb3_mu24.jsonl
echo "[$(date '+%F %T')] GLM LIVE FINAL DONE"
