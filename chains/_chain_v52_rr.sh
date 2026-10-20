#!/bin/bash
# v5.2 §6.5: run-to-run flip rate — identical config rerun (NoPool × GAIA × 24k)
cd /home/j0018/Desktop/tokenbank
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done
env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
BENCHMARK=GAIA BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=test \
V5=1 CONFIG_ONLY="V5.1-NoPool" OUT_TAG=_v52rr_ga TEMPERATURE=0 GAIA_N=1000 \
.venv/bin/python -u _budget_sweep_v2.py > logs/_v52rr_gaia.log 2>&1
