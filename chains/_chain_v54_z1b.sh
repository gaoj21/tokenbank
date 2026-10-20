#!/bin/bash
# v5.3 Z1b follow-up (§8.4b 遗留缺口): the two 400-task arms that close MATH attribution.
#   P1  Qwen  x V5.1-Ab6-NoM2NoM1 x MATH 24k  -- isolates M1 where it can actually act
#                                                (Ab3 - Ab6 = M1's own SR contribution)
#   P2  Gemma x V5-Unc-Fmt        x MATH 24k  -- the missing format-aligned baseline;
#                                                decides whether +25.8pp survives the
#                                                control that halved the Qwen number
# P2 needs llama-server on :8001 -> vLLM must be stopped first (see _chain_v54_p2.sh).
cd /home/j0018/Desktop/tokenbank
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done
env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=MATH BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=test \
  V5=1 CONFIG_ONLY="V5.1-Ab6-NoM2NoM1" OUT_TAG=_v54a_ma_ab6 TEMPERATURE=0 MATH_N=500 \
  .venv/bin/python -u _budget_sweep_v2.py > logs/_v54a_math_ab6.log 2>&1
