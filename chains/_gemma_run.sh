#!/bin/bash
# Gemma-4 with thinking: the pipeline samples with Gemma's recommended settings
# (temperature 1.0, top_p 0.95, top_k 64; greedy decoding loops in thinking mode), the cheap rungs
# are greedy with thinking off. GEMMA_MODEL / GEMMA_TAG select the checkpoint (default 31B dense).
# usage: _gemma_run.sh OUT_TAG [KEY=VALUE ...]
cd /home/j0018/Desktop/tokenbank
t=$1; shift
GM=${GEMMA_MODEL:-google/gemma-4-31b-it}; GT=${GEMMA_TAG:-gemma-4-31B-think}
[ -s "traces/tasks_sweep_$GT$t.jsonl" ] && { echo "SKIP $t"; exit 0; }
echo "[$(date '+%F %T')] START $t"
env OPENAI_BASE_URL=http://127.0.0.1:${PORT:-18000}/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=$GM \
    MODEL_TAG=$GT SEEDS=42 TEMPERATURE=1.0 L01_TEMPERATURE=0 L01_THINKING_OFF=1 V5=1 LOGPROBS_L01_ONLY=1 \
    MAX_WORKERS=${MAX_WORKERS:-256} MAX_TOKENS=4096 OUT_TAG=$t \
    LLM_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":true},"top_p":0.95,"top_k":64}' \
    "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$GT$t.log 2>&1
echo "[$(date '+%F %T')] DONE  $t"
