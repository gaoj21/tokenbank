#!/bin/bash
# Re-run the Ab7 arm on SWE with a config that matches the other two arms.
set -e
cd /home/j0018/Desktop/tokenbank
source ./_llm5_env.sh
echo "[$(date '+%F %T')] START sweo_ab7_fixed"
env BENCHMARK=SWEO SWE_N=500 BUDGETS=24000 SEEDS=42 MAX_WORKERS=$WORKERS SPLIT=full \
  V5=1 V5_MAX_ROUNDS=20 CONFIG_ONLY="SWEO-Ab7" OUT_TAG=_sweo_ab7fix \
  TEMPERATURE=0 DUMP_RAW_ANSWER=1 \
  .venv/bin/python -u _budget_sweep_v2.py > logs/_sweo_ab7fix.log 2>&1
echo "[$(date '+%F %T')] DONE sweo_ab7_fixed"
