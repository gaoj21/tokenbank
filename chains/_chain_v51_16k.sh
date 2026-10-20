#!/bin/bash
# v5.1 16k test-split results, all four datasets (with Unc baseline)
cd /home/j0018/Desktop/tokenbank
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done
run() {
  local bench=$1 tag=$2 log=$3; shift 3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=16000 SEEDS=42 MAX_WORKERS=64 SPLIT=test \
  V5=1 CONFIG_ONLY="V5.1-NoPool,V5.1-Unified,V5-Unconstrained" OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run GAIA    _v51t16_ga _v51test16_gaia.log    GAIA_N=1000
run MATH    _v51t16_ma _v51test16_math.log    MATH_N=500
run FRAMES  _v51t16_fr _v51test16_frames.log
run MUSIQUE _v51t16_mu _v51test16_musique.log MUSIQUE_N=99999
