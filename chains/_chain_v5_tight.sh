#!/bin/bash
# v5 tight-budget validation: M4/M1 only create value when the pool binds.
# Key configs at 8k+16k on test splits.
cd /home/j0018/Desktop/tokenbank
run() {
  local bench=$1 tag=$2 log=$3; shift 3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=8000,16000 SEEDS=42 MAX_WORKERS=64 SPLIT=test \
  V5=1 CONFIG_ONLY="V5-A0-Full,V5-A4-NoM4,V5-Unconstrained" OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run GAIA    _v5tight_ga _v5tight_gaia.log    GAIA_N=1000
run MUSIQUE _v5tight_mu _v5tight_musique.log MUSIQUE_N=99999
