#!/bin/bash
# FRAMES and MuSiQue for the reasoning-on Qwen family, queued behind the first chain so the
# two never contend for the same DashScope concurrency budget.
#
# The first chain took BigCodeBench, MATH and GAIA, which are the three that discriminate:
# BCB is where the reasoning rung is strongest on both thinking families, and MATH shows the
# gradient with the cap. These two complete the family so its row matches the other three.
set -e
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] waiting for the first chain (pid 1789465)"
while kill -0 1789465 2>/dev/null; do sleep 60; done
ARMS="V5-Unc-Fmt,V5.1-Ab7-NoVerbal,V5.2-ReasonLadder"
run() {
  local bench=$1 budget=$2 tag=$3 extra=$4
  [ -s "tasks_sweep_Qwen3.8-flash-think$tag.jsonl" ] && { echo "SKIP $tag"; return; }
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
    VLLM_API_KEY="$(cat /home/j0018/.config/dashscope.key)" \
    VLLM_MODEL=qwen3.8-flash MODEL_TAG=Qwen3.8-flash-think \
    LLM_EXTRA_BODY_JSON='{"enable_thinking":true}' \
    BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=96 SPLIT=test \
    V5=1 CONFIG_ONLY="$ARMS" OUT_TAG=$tag TEMPERATURE=0 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
for B in 24000 16000 8000; do run FRAMES  $B _qt_fr${B%000} x=1; done
for B in 24000 16000 8000; do run MUSIQUE $B _qt_mu${B%000} MUSIQUE_N=99999; done
echo "[$(date '+%F %T')] QWEN THINKING FRAMES+MUSIQUE DONE"
