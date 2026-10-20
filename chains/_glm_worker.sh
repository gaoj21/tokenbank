#!/bin/bash
# A worker that waits for its endpoint to be free, then pulls BCB budgets from a shared
# queue until the queue is empty.
#
# Claiming is `mkdir`, which is atomic on a local filesystem: exactly one worker can create
# a given directory, so two endpoints can never take the same budget. This matters because a
# duplicate claim would have both of them appending to one trace file, which is the failure
# that produced a cell holding two copies of every task earlier in this project.
#
# Usage: _glm_worker.sh <url> <tag> [wait-for-pid]
set -u
cd /home/j0018/Desktop/tokenbank
URL=$1; TAG=$2; WAITPID=${3:-}
if [ -n "$WAITPID" ]; then
  echo "[$(date '+%F %T')] $TAG waiting for pid $WAITPID"
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 30; done
fi
while true; do
  claimed=""
  for b in 4000 8000 16000; do
    tag=_glmbcb${b%000}k
    [ -s "tasks_sweep_GLM-5.3-Flash$tag.jsonl" ] && continue
    if mkdir ".glmq/$tag" 2>/dev/null; then claimed=$b; break; fi
  done
  [ -z "$claimed" ] && { echo "[$(date '+%F %T')] $TAG: queue empty"; break; }
  tag=_glmbcb${claimed%000}k
  echo "[$(date '+%F %T')] $TAG claimed $tag"
  env OPENAI_BASE_URL="$URL" VLLM_API_KEY=EMPTY \
    VLLM_MODEL=zai-org/GLM-5.3-Flash MODEL_TAG=GLM-5.3-Flash \
    BENCHMARK=BCB BCB_N=500 BUDGETS=$claimed SEEDS=42 MAX_WORKERS=32 SPLIT=full \
    V5=1 V5_MAX_ROUNDS=5 CONFIG_ONLY="BCB-Naive,BCB-Ab7,BCB-ReasonLadder" \
    OUT_TAG=$tag TEMPERATURE=0 DUMP_RAW_ANSWER=1 \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] $TAG finished $tag"
done
