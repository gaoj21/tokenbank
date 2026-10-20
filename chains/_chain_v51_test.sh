#!/bin/bash
# v5.1 final test-split results, all four datasets @24k
cd /home/j0018/Desktop/tokenbank
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done
run() {
  local bench=$1 tag=$2 log=$3; shift 3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=test \
  V5=1 CONFIG_ONLY="V5.1-NoPool,V5.1-Unified" OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run GAIA    _v51t_ga _v51test_gaia.log    GAIA_N=1000
run MATH    _v51t_ma _v51test_math.log    MATH_N=500
run FRAMES  _v51t_fr _v51test_frames.log
run MUSIQUE _v51t_mu _v51test_musique.log MUSIQUE_N=99999
