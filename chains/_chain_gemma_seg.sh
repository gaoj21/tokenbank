#!/bin/bash
# Segment accounting on Gemma: the paper's motivating measurement, which slot the budget
# goes to, currently exists for one family only. Same configuration as the Qwen run it is
# compared against: the untreated format-aligned pipeline at 16k, with SEGMENT_ACCOUNTING on.
set -e
cd /home/j0018/Desktop/tokenbank
KEY=$(cat /home/j0018/.config/openrouter.key)
for pair in "GAIA ga GAIA_N=1000" "MATH ma MATH_N=500" "FRAMES fr x=1" "MUSIQUE mu MUSIQUE_N=99999" "BCB bcb BCB_N=500"; do
  set -- $pair; bench=$1; tag=$2; extra=$3
  out=_gseg_$tag
  [ -s "tasks_sweep_gemma-4-31B-bf16$out.jsonl" ] && { echo "SKIP $out"; continue; }
  arm="V5-Unc-Fmt"; [ "$bench" = "BCB" ] && arm="BCB-Naive"
  echo "[$(date '+%F %T')] START $out"
  env OPENAI_BASE_URL=https://openrouter.ai/api/v1 VLLM_API_KEY="$KEY" \
    VLLM_MODEL=google/gemma-4-31b-it MODEL_TAG=gemma-4-31B-bf16 \
    LLM_EXTRA_BODY_JSON='{"provider":{"only":["venice"],"allow_fallbacks":false}}' \
    BENCHMARK=$bench BUDGETS=16000 SEEDS=42 MAX_WORKERS=64 SPLIT=test \
    V5=1 CONFIG_ONLY="$arm" OUT_TAG=$out TEMPERATURE=0 SEGMENT_ACCOUNTING=1 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$out.log 2>&1
  echo "[$(date '+%F %T')] DONE  $out"
done
echo "[$(date '+%F %T')] GEMMA SEG DONE"
