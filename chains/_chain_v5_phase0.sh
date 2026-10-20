#!/bin/bash
# v5 Phase 0: re-run Unconstrained + Unc+Compact with per-round instrumentation
# and logprob capture, on GAIA and MuSiQue (full).
cd /home/j0018/Desktop/tokenbank
run() {
  local bench=$1 tag=$2 log=$3; shift 3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=16000 SEEDS=42 MAX_WORKERS=64 \
  CONFIG_ONLY="Unconstrained,Unc+Compact" OUT_TAG=$tag TEMPERATURE=0 CAPTURE_LOGPROBS=1 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run GAIA    _v5diag_ga _v5diag_gaia.log    GAIA_N=1000
run MUSIQUE _v5diag_mu _v5diag_musique.log MUSIQUE_N=99999
