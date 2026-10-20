#!/bin/bash
# wait until the Improved+Compact FRAMES runs (16K then 24K) are all done,
# then run the baseline configs on FRAMES.
cd /home/j0018/Desktop/tokenbank
while pgrep -f "_budget_sweep_v2.py" > /dev/null; do sleep 60; done
OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
BENCHMARK=FRAMES BUDGETS=16000 SEEDS=42 MAX_WORKERS=64 \
CONFIG_ONLY="Unconstrained,Static Quota" OUT_TAG=_fr16k_base TEMPERATURE=0 \
.venv/bin/python -u _budget_sweep_v2.py > logs/_frames16k_baselines.log 2>&1
OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
BENCHMARK=FRAMES BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 \
CONFIG_ONLY="Static Quota" OUT_TAG=_fr24k_base TEMPERATURE=0 \
.venv/bin/python -u _budget_sweep_v2.py > logs/_frames24k_baselines.log 2>&1
