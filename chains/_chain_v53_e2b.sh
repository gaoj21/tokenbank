#!/bin/bash
# v5.3 E2b: MuSiQue attribution — the two remaining untested arms after both E2 arms came back null.
# Unc+Fmt (format test, C1 completion) and -M2 (cascade+verbal removal, keep M1+M3).
cd /home/j0018/Desktop/tokenbank
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done
run() {
  local cfgs=$1 tag=$2 log=$3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=MUSIQUE BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=test \
  V5=1 CONFIG_ONLY="$cfgs" OUT_TAG=$tag TEMPERATURE=0 MUSIQUE_N=99999 \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run "V5.1-Ab3-NoM2" _v53e2b_mu_nom2 _v53e2b_mu_nom2.log
run "V5-Unc-Fmt"    _v53e2b_mu_fmt  _v53e2b_mu_fmt.log
