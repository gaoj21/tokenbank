#!/bin/bash
# The one asymmetry between the two reasoning families.
#
# GLM has an Ab7 cell on MATH, which is what prices the reasoning rung against prefix-only
# governance: Ab7 bounds the carried slots and leaves per-call generation alone, so the gap
# between it and the ladder is the part of the result that the reasoning rung is responsible
# for. The DeepSeek chain ran three arms and has that comparison on BigCodeBench only, where
# it is stark (Ab7 collapses onto Naive, 18% non-empty answers against 18%). MATH is the
# harder case for the claim because the task needs few rounds, so it is the cell worth having.
set -e
cd /home/j0018/Desktop/tokenbank
TAG=DeepSeek-V4.1-Flash
for B in 24000 16000 8000; do
  t=_dsab7_ma${B%000}
  [ -s "tasks_sweep_$TAG$t.jsonl" ] && { echo "SKIP $t"; continue; }
  echo "[$(date '+%F %T')] START $t"
  env OPENAI_BASE_URL=http://127.0.0.1:18000/v1 VLLM_API_KEY=EMPTY \
      VLLM_MODEL=deepseek-v4.1-flash MODEL_TAG=$TAG SEEDS=42 MAX_WORKERS=256 \
      TEMPERATURE=0 V5=1 BENCHMARK=MATH MATH_N=500 BUDGETS=$B SPLIT=test \
      CONFIG_ONLY=V5.1-Ab7-NoVerbal OUT_TAG=$t \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"
done
echo "[$(date '+%F %T')] DS AB7 DONE"
