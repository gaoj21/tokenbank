#!/bin/bash
# Gemma-4-26B-A4B-it, standard checkpoint at bf16 on the local servers, with the settings of the Makora run
# (_gemma_mk_run.sh): pipeline thinking on with Gemma's sampling (temperature 1.0, top_p 0.95, top_k 64), cheap
# rungs greedy with thinking off. Records are tagged gemma-4-26B-A4B-bf16.
# usage: PORT=... _gemma_local_run.sh OUT_TAG [KEY=VALUE ...]
cd /home/j0018/Desktop/tokenbank
t=$1; shift; GT=gemma-4-26B-A4B-bf16
[ -s "traces/tasks_sweep_$GT$t.jsonl" ] && { echo "SKIP $t"; exit 0; }
echo "[$(date '+%F %T')] START $t"
env OPENAI_BASE_URL=http://127.0.0.1:${PORT}/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=google/gemma-4-26b-a4b-it \
    MODEL_TAG=$GT SEEDS=42 TEMPERATURE=1.0 L01_TEMPERATURE=0 L01_THINKING_OFF=1 V5=1 LOGPROBS_L01_ONLY=1 \
    MAX_WORKERS=${MAX_WORKERS:-64} MAX_TOKENS=4096 OUT_TAG=$t \
    LLM_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":true},"top_p":0.95,"top_k":64}' \
    "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$GT$t.log 2>&1
echo "[$(date '+%F %T')] DONE  $t"
