#!/bin/bash
# v5 dev-split calibration data: run A0 with all stops disabled (V5_CALIB=1)
# so every level+round is observed; thresholds are tuned OFFLINE from traces.
cd /home/j0018/Desktop/tokenbank
run() {
  local bench=$1 tag=$2 log=$3; shift 3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=16000 SEEDS=42 MAX_WORKERS=64 SPLIT=dev \
  V5=1 V5_CALIB=1 CONFIG_ONLY="V5-A0-Full" OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run GAIA    _v5cal_ga _v5cal_gaia.log    GAIA_N=1000
run MUSIQUE _v5cal_mu _v5cal_musique.log MUSIQUE_N=99999
