#!/bin/bash
cd /home/j0018/Desktop/tokenbank
source ./_llm5_env.sh
env BENCHMARK=MATH BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=test MATH_N=500 \
    V5=1 CONFIG_ONLY="V5-Unc-Fmt" OUT_TAG=_spd TEMPERATURE=0 \
    .venv/bin/python -u _budget_sweep_v2.py > logs/_spd.log 2>&1
