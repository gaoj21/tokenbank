#!/bin/bash
# Calibration run for the confidence-gate transfer table: no early stop, thresholds at
# infinity, so every level and round is observed and offline threshold sweeps can replay
# the traces. The original Qwen calibration traces are not on disk any more, so this runs
# both families under one procedure rather than comparing against a number we cannot
# reproduce.
set -e
cd /home/j0018/Desktop/tokenbank
KEY=$(cat /home/j0018/.config/openrouter.key)
for pair in "GAIA ga GAIA_N=1000" "MATH ma MATH_N=500" "FRAMES fr x=1" "MUSIQUE mu MUSIQUE_N=99999"; do
  set -- $pair; bench=$1; tag=$2; extra=$3
  out=_gcal_$tag
  [ -s "tasks_sweep_gemma-4-31B-bf16$out.jsonl" ] && { echo "SKIP $out"; continue; }
  echo "[$(date '+%F %T')] START $out"
  env OPENAI_BASE_URL=https://openrouter.ai/api/v1 VLLM_API_KEY="$KEY" \
    VLLM_MODEL=google/gemma-4-31b-it MODEL_TAG=gemma-4-31B-bf16 \
    LLM_EXTRA_BODY_JSON='{"provider":{"only":["venice"],"allow_fallbacks":false}}' \
    BENCHMARK=$bench BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=test \
    V5=1 V5_CALIB=1 CONFIG_ONLY="V5.1-Ab7-NoVerbal" OUT_TAG=$out TEMPERATURE=0 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$out.log 2>&1
  echo "[$(date '+%F %T')] DONE  $out"
done
echo "[$(date '+%F %T')] GEMMA CALIB DONE"
