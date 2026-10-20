#!/bin/bash
# wait for the Combo16k run to finish, then run the same config at 24K
while kill -0 3630453 2>/dev/null; do sleep 30; done
cd /home/j0018/Desktop/tokenbank
OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
BENCHMARK=GAIA GAIA_N=1000 BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 \
CONFIG_ONLY="Combo16k" OUT_TAG=_ga24k_combo TEMPERATURE=0 \
.venv/bin/python -u _budget_sweep_v2.py > logs/_gaia24k_combo.log 2>&1
