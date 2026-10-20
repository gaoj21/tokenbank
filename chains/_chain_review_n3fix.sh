#!/bin/bash
# Re-run the N3 tasks whose call failed on the llm2 endpoint (Cloudflare 524), on stable FP8 endpoints.
cd /home/j0018/Desktop/tokenbank
G=GLM-5.3-Flash; M=GLM-5.3-Flash-FP8; I=results/climb_live
run() { local t=$1 url=$2; shift 2
  echo "[$(date '+%F %T')] START $t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$M MODEL_TAG=$G SEEDS=42 TEMPERATURE=0 V5=1 MAX_TOKENS=16384 \
    OUT_TAG=$t BUDGETS=24000 SPLIT=test CONFIG_ONLY=V5-Single-Fmt-16k "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$G$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
L3=https://llm3.ruoyu-wang.com/v1; L5=https://llm5.ruoyu-wang.com/v1
( run _n3fix_ma24 $L3 BENCHMARK=MATH MATH_N=500 TASK_IDS_FILE=$I/n3_fix_ids_ma.json
  run _n3fix_ga24 $L3 BENCHMARK=GAIA GAIA_N=1000 TASK_IDS_FILE=$I/n3_fix_ids_ga.json
  run _n3fix_fr24 $L3 BENCHMARK=FRAMES TASK_IDS_FILE=$I/n3_fix_ids_fr.json ) &
( run _n3fix_mu24 $L5 BENCHMARK=MUSIQUE MUSIQUE_N=99999 TASK_IDS_FILE=$I/n3_fix_ids_mu.json ) &
wait; echo "[$(date '+%F %T')] N3 FIX DONE"
