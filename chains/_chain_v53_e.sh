#!/bin/bash
# v5.3 Phase E (ungated): E1 MATH×Unc+Fmt -> E3 8k third budget point -> E4 MATH flip-rate rerun
cd /home/j0018/Desktop/tokenbank
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done
run() {
  local bench=$1 budget=$2 cfgs=$3 tag=$4 log=$5; shift 5
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=64 SPLIT=test \
  V5=1 CONFIG_ONLY="$cfgs" OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
# E1: MATH x Unc+Fmt (hole A — the decisive control for the biggest claim)
run MATH 24000 "V5-Unc-Fmt" _v53e1_ma24 _v53e1_math24.log MATH_N=500
run MATH 16000 "V5-Unc-Fmt" _v53e1_ma16 _v53e1_math16.log MATH_N=500
# E3: third budget point 8k on GAIA+FRAMES (hole C — same-frontier test)
run GAIA   8000 "V5.1-NoPool,V5-Unc-Fmt" _v53e3_ga8 _v53e3_gaia8.log GAIA_N=1000
run FRAMES 8000 "V5.1-NoPool,V5-Unc-Fmt" _v53e3_fr8 _v53e3_frames8.log
# E4: MATH flip-rate rerun (same config twice)
run MATH 24000 "V5.1-NoPool" _v53e4_ma _v53e4_math.log MATH_N=500
