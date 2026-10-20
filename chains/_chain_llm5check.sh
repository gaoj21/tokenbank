#!/bin/bash
# Before mixing llm5 results with the existing table, check that it is the same
# model in effect. llm9 served Qwen3.8-27B with runtime FP8 quantization; llm5
# serves a pre-quantized FP8 checkpoint, which is not numerically identical.
# Re-run one cell that already exists (MATH 24k, format-aligned baseline, 57.2%)
# and compare per item. Agreement within the measured run-to-run flip rate for
# this model (9.5% on MATH) means the two are interchangeable; materially worse
# agreement means the existing table cannot be extended with llm5 runs.
set -e
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] START llm5check"
env OPENAI_BASE_URL=https://llm5.ruoyu-wang.com/v1 VLLM_API_KEY=EMPTY \
  VLLM_MODEL=qwen3.8-27b-fp8 MODEL_TAG=Qwen3.8-27B \
  BENCHMARK=MATH BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=test MATH_N=500 \
  V5=1 CONFIG_ONLY="V5-Unc-Fmt" OUT_TAG=_llm5chk TEMPERATURE=0 \
  .venv/bin/python -u _budget_sweep_v2.py > logs/_llm5chk.log 2>&1
echo "[$(date '+%F %T')] DONE llm5check"
