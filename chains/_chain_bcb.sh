#!/bin/bash
# BigCodeBench, three arms across a budget range chosen to span the point where
# the constraint starts to bind.
#
# Budgets are 4k/8k/16k rather than the 8k/16k/24k used elsewhere: at 16k only
# 5% of runs exhaust the budget and 452 of 500 converge in a single round, so
# the multi-round loop this work is about barely engages. 4k puts 35% of runs
# over the cap, 8k puts 8.6% over, and 16k is the non-binding control. If the
# thesis is right the arms should separate at 4k and converge at 16k.
#
# Runs serially: the server saturates around 32-40 concurrent requests, so one
# job at 32 workers finishes faster than three jobs sharing the same server.
set -e
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] BCB waiting for the running sweep"
while pgrep -f "_chain_ab7_full" >/dev/null; do sleep 60; done
for b in 4000 8000 16000; do
  tag="_bcb${b%000}k"
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
    VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
    BENCHMARK=BCB BCB_N=500 BUDGETS=$b SEEDS=42 MAX_WORKERS=32 SPLIT=full \
    V5=1 V5_MAX_ROUNDS=5 CONFIG_ONLY="BCB-Naive,BCB-Ours,BCB-Ab7" \
    OUT_TAG=$tag TEMPERATURE=0 DUMP_RAW_ANSWER=1 \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
done
echo "[$(date '+%F %T')] BCB ALL DONE"
