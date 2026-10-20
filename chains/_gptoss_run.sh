#!/bin/bash
# gpt-oss with the same ladder as GLM: cheap rungs at reasoning_effort=low and greedy (CLIMB-Live config),
# the pipeline samples at temperature 1.0 (greedy decoding degenerates at medium effort),
# pipeline at the model's default (medium). Log-probs only on the calls that score the cheap rungs.
# usage: PORT=... MODEL=gpt-oss-20b _gptoss_run.sh OUT_TAG [KEY=VALUE ...]
cd /home/j0018/Desktop/tokenbank
M=${MODEL:-gpt-oss-20b}; t=$1; shift
[ -s "traces/tasks_sweep_$M$t.jsonl" ] && { echo "SKIP $t"; exit 0; }
echo "[$(date '+%F %T')] START $t"
env OPENAI_BASE_URL=http://127.0.0.1:${PORT:-18101}/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=openai/$M \
    MODEL_TAG=$M SEEDS=42 TEMPERATURE=${TEMPERATURE:-1.0} L01_TEMPERATURE=0 V5=1 LOGPROBS_L01_ONLY=1 TOP_LOGPROBS=0 DECLARE_PY_TOOL=1 \
    MAX_WORKERS=${MAX_WORKERS:-128} MAX_TOKENS=4096 OUT_TAG=$t \
    "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$M$t.log 2>&1
echo "[$(date '+%F %T')] DONE  $t"
