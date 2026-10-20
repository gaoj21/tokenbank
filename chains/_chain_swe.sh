#!/bin/bash
# C-minimal: long-horizon compaction comparison on SWE-bench Lite.
# Four arms at a matched 24k budget and a 20-round cap:
#   Ours            -- rebuild each round, every carried slot capped
#   Compact-Real    -- accumulate, LLM-summarize at 0.5 of quota (aggressive)
#   Compact-Late    -- same but trigger at 0.8 (conservative)
#   Naive+Fmt       -- rebuild each round, slots unbounded (the paper's baseline)
# Judging is the token-F1 proxy, NOT official SWE-bench test execution.
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] START swe_full"
env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
  VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
  BENCHMARK=SWE SWE_N=300 BUDGETS=24000 SEEDS=42 MAX_WORKERS=48 SPLIT=full \
  V5_MAX_ROUNDS=20 V5=1 \
  CONFIG_ONLY="V5.1-NoPool,V5-Compact-Real,V5-Compact-Late,V5-Unc-Fmt" \
  OUT_TAG=_swe300 TEMPERATURE=0 \
  .venv/bin/python -u _budget_sweep_v2.py > logs/_swe300.log 2>&1
echo "[$(date '+%F %T')] DONE swe_full"
