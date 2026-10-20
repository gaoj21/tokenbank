#!/bin/bash
# One BCB shard. Usage: _chain_glm_bcb_shard.sh <url> <tag> <budgets...>
#
# Arms follow the BCB convention, not the main table's: the answer is a code block judged by
# executing the task's own unit tests, so the ANSWER: format alignment that Naive+Fmt exists
# to control for does not apply. BCB-Ab7 is kept here even though the main table dropped it,
# because on code it is the arm that prices the reasoning rung; the main table has the GAIA
# cells for that and BCB does not.
#
# llm56 and llm57 are the same deployment as llm52: same model_path and tp_size, and with
# reasoning_effort=low they return token-identical generations on a probe prompt. Each
# budget is written by exactly one shard.
set -eu
cd /home/j0018/Desktop/tokenbank
URL=$1; TAG=$2; shift 2
for b in "$@"; do
  tag=_glmbcb${b%000}k
  [ -s "tasks_sweep_GLM-5.3-Flash$tag.jsonl" ] && { echo "SKIP $tag"; continue; }
  echo "[$(date '+%F %T')] START $tag on $TAG"
  env OPENAI_BASE_URL="$URL" VLLM_API_KEY=EMPTY \
    VLLM_MODEL=zai-org/GLM-5.3-Flash MODEL_TAG=GLM-5.3-Flash \
    BENCHMARK=BCB BCB_N=500 BUDGETS=$b SEEDS=42 MAX_WORKERS=32 SPLIT=full \
    V5=1 V5_MAX_ROUNDS=5 CONFIG_ONLY="BCB-Naive,BCB-Ab7,BCB-ReasonLadder" \
    OUT_TAG=$tag TEMPERATURE=0 DUMP_RAW_ANSWER=1 \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag on $TAG"
done
echo "[$(date '+%F %T')] BCB SHARD $TAG DONE"
