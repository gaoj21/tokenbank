#!/bin/bash
# v5 Phase 3.1 ablation on dev split @24k (plan specifies 24K-equivalent)
cd /home/j0018/Desktop/tokenbank
run() {
  local bench=$1 tag=$2 log=$3; shift 3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=dev \
  V5=1 OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run GAIA    _v5abl_ga _v5abl_gaia.log    GAIA_N=1000
run MUSIQUE _v5abl_mu _v5abl_musique.log MUSIQUE_N=99999
