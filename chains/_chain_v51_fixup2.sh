#!/bin/bash
# rerun GAIA dev for V5.1-Unified after the ESCALATE pooling fix
cd /home/j0018/Desktop/tokenbank
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done
env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
BENCHMARK=GAIA BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=dev \
V5=1 CONFIG_ONLY="V5.1-Unified" OUT_TAG=_v51gafix TEMPERATURE=0 GAIA_N=1000 \
.venv/bin/python -u _budget_sweep_v2.py > logs/_v51_gaia_fix.log 2>&1
