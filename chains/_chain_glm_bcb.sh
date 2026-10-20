#!/bin/bash
# BigCodeBench on GLM-5.3-Flash, on the endpoint that finished its shard first.
#
# Arms follow the BCB convention rather than the main table's: the answer is a code block
# judged by executing the task's own unit tests, so the ANSWER: format alignment that
# Naive+Fmt controls for does not apply. BCB-ReasonLadder stands to BCB-Ab7 exactly as
# V5.2-ReasonLadder stands to V5.1-Ab7-NoVerbal, so the pair prices the reasoning rung on
# code the same way the main table prices it elsewhere.
# DUMP_RAW_ANSWER=1 is required; the sandboxed judge runs offline afterwards.
set -e
cd /home/j0018/Desktop/tokenbank
for b in 4000 8000 16000; do
  tag=_glmbcb${b%000}k
  [ -s "tasks_sweep_GLM-5.3-Flash$tag.jsonl" ] && { echo "SKIP $tag"; continue; }
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://llm53.jierungogogo.com/v1 VLLM_API_KEY=EMPTY \
    VLLM_MODEL=zai-org/GLM-5.3-Flash MODEL_TAG=GLM-5.3-Flash \
    BENCHMARK=BCB BCB_N=500 BUDGETS=$b SEEDS=42 MAX_WORKERS=32 SPLIT=full \
    V5=1 V5_MAX_ROUNDS=5 CONFIG_ONLY="BCB-Naive,BCB-Ab7,BCB-ReasonLadder" \
    OUT_TAG=$tag TEMPERATURE=0 DUMP_RAW_ANSWER=1 \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
done
echo "[$(date '+%F %T')] GLM BCB DONE (judge offline with _bcb_rejudge.py)"
