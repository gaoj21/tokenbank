#!/bin/bash
# Compaction baseline, re-run with the corrected trigger.
# The first run's `kv_trigger` compared an agent's CUMULATIVE SPEND against its
# budget quota, so it fired on exactly the items already headed for exhaustion
# (100% of triggered items truncated). It is now context occupancy:
#   Compact-Real   -- 8k window, 0.5 => fires at 4,096 prompt tokens.
#                     Measured peak prompt on SWE is 4,067, so this should
#                     essentially never fire. That null IS the result: under a
#                     hard token budget the budget binds before the window does.
#   Compact-Forced -- 2k window, 0.5 => fires at 1,024, below the median prompt
#                     (1,818), so compaction actually operates and can be priced.
# Ours and Naive+Fmt are unchanged from _swe300 and are not re-run.
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] START swe_compactfix"
env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
  VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
  BENCHMARK=SWE SWE_N=300 BUDGETS=24000 SEEDS=42 MAX_WORKERS=32 SPLIT=full \
  V5_MAX_ROUNDS=20 V5=1 \
  CONFIG_ONLY="V5-Compact-Real,V5-Compact-Forced" \
  OUT_TAG=_swecmp2 TEMPERATURE=0 \
  .venv/bin/python -u _budget_sweep_v2.py > logs/_swecmp2.log 2>&1
echo "[$(date '+%F %T')] DONE swe_compactfix"
