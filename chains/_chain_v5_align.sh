#!/bin/bash
# v5 alignment on MATH-500 and FRAMES: recommended-form configs vs upper bound.
# No dev tuning was ever done on these datasets -> run full split.
cd /home/j0018/Desktop/tokenbank
run() {
  local bench=$1 tag=$2 log=$3; shift 3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=full \
  V5=1 CONFIG_ONLY="V5-A0-Full,V5-A4-NoM4,V5-A6-OnlyM3,V5-Unconstrained" \
  OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run MATH   _v5al_ma _v5al_math.log   MATH_N=500
run FRAMES _v5al_fr _v5al_frames.log
