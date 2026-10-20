#!/bin/bash
# v5.3 E2: MuSiQue attribution ablation @24k — gated by Z1 (verdict: run both arms).
# Waits for any running sweep (Phase E chain) before starting.
cd /home/j0018/Desktop/tokenbank
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done
run() {
  local cfgs=$1 tag=$2 log=$3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=MUSIQUE BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=test \
  V5=1 CONFIG_ONLY="$cfgs" OUT_TAG=$tag TEMPERATURE=0 MUSIQUE_N=99999 \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run "V5.1-Ab1-NoM1"    _v53e2_mu_nom1   _v53e2_mu_nom1.log
run "V5.1-Ab5-NoM3ctx" _v53e2_mu_nom3   _v53e2_mu_nom3.log
