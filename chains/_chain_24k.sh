#!/bin/bash
# wait for the 16K compact run to finish, then launch the 24K sweep
while kill -0 3534338 2>/dev/null; do sleep 30; done
cd /home/j0018/Desktop/tokenbank
OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
BENCHMARK=GAIA GAIA_N=1000 BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 \
CONFIG_ONLY="Static Quota,TB-v2-Full,Full+Stall,Improved,Compact,Improved+Compact" \
OUT_TAG=_ga24k TEMPERATURE=0 \
.venv/bin/python -u _budget_sweep_v2.py > logs/_gaia24k.log 2>&1
