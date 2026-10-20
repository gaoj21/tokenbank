#!/bin/bash
# The ablation that prices the reasoning rung, on the benchmark that can resolve it.
#
# V5.1-Ab7-NoVerbal is the configuration used for the two families whose thinking is off:
# bounded slots, convergence stop, log-probability routing, and nothing governing the
# per-call generation. V5.2-ReasonLadder is that arm plus a cheaper reasoning setting on the
# levels the gate keeps. The difference between them is the whole claim of this run, and it
# currently rests on GAIA at 24k and 16k, which is 122 items and the weakest of the four.
# MATH has 400 items and the cleanest judging, so the pair is worth having there.
#
# Only the Ab7 arm is run; ReasonLadder and the baselines are already on disk under
# _glm4_ma{8,16,24}, from the same code version.
set -e
cd /home/j0018/Desktop/tokenbank
for b in 24000 16000 8000; do
  tag=_glmab7_ma${b%000}
  [ -s "tasks_sweep_GLM-5.3-Flash$tag.jsonl" ] && { echo "SKIP $tag"; continue; }
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://llm56.jierungogogo.com/v1 VLLM_API_KEY=EMPTY \
    VLLM_MODEL=zai-org/GLM-5.3-Flash MODEL_TAG=GLM-5.3-Flash \
    BENCHMARK=MATH MATH_N=500 BUDGETS=$b SEEDS=42 MAX_WORKERS=32 SPLIT=test \
    V5=1 CONFIG_ONLY="V5.1-Ab7-NoVerbal" OUT_TAG=$tag TEMPERATURE=0 \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
done
echo "[$(date '+%F %T')] AB7 MATH DONE"
