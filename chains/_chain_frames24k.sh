#!/bin/bash
while kill -0 3701633 2>/dev/null; do sleep 60; done
cd /home/j0018/Desktop/tokenbank
OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
BENCHMARK=FRAMES BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 \
CONFIG_ONLY="Improved+Compact" OUT_TAG=_fr24k TEMPERATURE=0 \
.venv/bin/python -u _budget_sweep_v2.py > logs/_frames24k_ic.log 2>&1
