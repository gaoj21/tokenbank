#!/bin/bash
cd /home/j0018/Desktop/tokenbank
run() {  # bench budgets configs tag log [extra_env...]
  local bench=$1 budgets=$2 configs=$3 tag=$4 log=$5; shift 5
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=$budgets SEEDS=42 MAX_WORKERS=64 CONFIG_ONLY="$configs" \
  OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
# 32K for the three finished datasets (Unconstrained is budget-independent, reused)
run GAIA   32000 "Static Quota,Improved+Compact" _ga32k _gaia32k.log
run FRAMES 32000 "Static Quota,Improved+Compact" _fr32k _frames32k.log
run MATH   32000 "Static Quota,Improved+Compact" _ma32k _math32k.log
# then MuSiQue full, aligned 16K/24K/32K
run MUSIQUE 16000 "Unconstrained,Static Quota,Improved+Compact" _mu16k _musique16k.log MUSIQUE_N=99999
run MUSIQUE 24000 "Static Quota,Improved+Compact"               _mu24k _musique24k.log MUSIQUE_N=99999
run MUSIQUE 32000 "Static Quota,Improved+Compact"               _mu32k _musique32k.log MUSIQUE_N=99999
