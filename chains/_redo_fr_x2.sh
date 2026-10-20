#!/bin/bash
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] START _climbx_fr24x2 (llm4, redo)"
env OPENAI_BASE_URL=https://llm4.ruoyu-wang.com/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=GLM-5.3-Flash-FP8 MODEL_TAG=GLM-5.3-Flash SEEDS=42 TEMPERATURE=0 V5=1 \
  MAX_WORKERS=32 MAX_TOKENS=4096 OUT_TAG=_climbx_fr24x2 CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 TASK_IDS_FILE=results/climb_live/fr_x2.json \
  BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test .venv/bin/python -u _budget_sweep_v2.py > logs/GLM-5.3-Flash_climbx_fr24x2.log 2>&1
echo "[$(date '+%F %T')] DONE  _climbx_fr24x2 APIERR=$(grep -c APIERR logs/GLM-5.3-Flash_climbx_fr24x2.log)"
