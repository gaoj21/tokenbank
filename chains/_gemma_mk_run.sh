#!/bin/bash
# Gemma-4-26B-A4B-it (standard checkpoint) through OpenRouter, pinned to the Makora provider. Same settings as
# the local QAT run: pipeline thinking on with Gemma's sampling (temperature 1.0, top_p 0.95, top_k 64), cheap
# rungs greedy with thinking off (reasoning.enabled=false). Records are tagged gemma-4-26B-A4B-mk, never -qat.
# usage: _gemma_mk_run.sh OUT_TAG [KEY=VALUE ...]
cd /home/j0018/Desktop/tokenbank
set -a; . /home/j0018/.config/tokenbank/openrouter.env; set +a
t=$1; shift; GT=gemma-4-26B-A4B-mk
[ -s "traces/tasks_sweep_$GT$t.jsonl" ] && { echo "SKIP $t"; exit 0; }
echo "[$(date '+%F %T')] START $t"
env OPENAI_BASE_URL=https://openrouter.ai/api/v1 VLLM_API_KEY="$OPENROUTER_API_KEY" VLLM_MODEL=google/gemma-4-26b-a4b-it \
    MODEL_TAG=$GT SEEDS=42 TEMPERATURE=1.0 L01_TEMPERATURE=0 L01_THINKING_OFF=1 OR_REASONING_SWITCH=1 V5=1 LOGPROBS_L01_ONLY=1 \
    MAX_WORKERS=${MAX_WORKERS:-24} MAX_TOKENS=4096 OUT_TAG=$t LLM_MAX_RETRIES=60 \
    LLM_EXTRA_BODY_JSON='{"provider":{"order":["Makora"],"allow_fallbacks":false,"require_parameters":true},"reasoning":{"enabled":true},"top_p":0.95,"top_k":64}' \
    "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$GT$t.log 2>&1
echo "[$(date '+%F %T')] DONE  $t"
