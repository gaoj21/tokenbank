#!/bin/bash
# v5.2 C1: Unc+Fmt control arm — GAIA + FRAMES @ 24k and 16k (I1 decisive test)
cd /home/j0018/Desktop/tokenbank
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done
run() {
  local bench=$1 budget=$2 tag=$3 log=$4; shift 4
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=64 SPLIT=test \
  V5=1 CONFIG_ONLY="V5-Unc-Fmt" OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run GAIA   24000 _v52c1_ga24 _v52c1_gaia24.log   GAIA_N=1000
run GAIA   16000 _v52c1_ga16 _v52c1_gaia16.log   GAIA_N=1000
run FRAMES 24000 _v52c1_fr24 _v52c1_frames24.log
run FRAMES 16000 _v52c1_fr16 _v52c1_frames16.log
