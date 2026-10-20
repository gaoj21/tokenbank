#!/bin/bash
# SWE-bench Verified, oracle localization, 500 instances x 2 arms.
# Generates the replies; the diffs are assembled locally afterwards and scored
# by the official harness (FAIL_TO_PASS / PASS_TO_PASS), not by any proxy.
# DUMP_RAW_ANSWER keeps the whole reply: the answer is a set of code regions,
# which the usual answer extraction would reduce to its last line.
# SPLIT=full, not test: the 80/20 dev holdout exists so mechanisms tuned on dev
# are not scored on the same items, and nothing was tuned on SWE. Taking 400 of
# the 500 would also make these incomparable to published Verified numbers.
set -e
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] START sweo"
env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
  VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
  BENCHMARK=SWEO SWE_N=500 BUDGETS=24000 SEEDS=42 MAX_WORKERS=24 SPLIT=full \
  V5=1 V5_MAX_ROUNDS=20 CONFIG_ONLY="SWEO-Naive,SWEO-Ours" \
  OUT_TAG=_sweo TEMPERATURE=0 DUMP_RAW_ANSWER=1 \
  .venv/bin/python -u _budget_sweep_v2.py > logs/_sweo.log 2>&1
echo "[$(date '+%F %T')] DONE sweo"
