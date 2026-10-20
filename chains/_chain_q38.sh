#!/bin/bash
# Qwen3.8-27B (remote vLLM) as the new primary model.
# Runs on separate hardware from the local box, so it does not contend with the
# v5.5 chain still finishing MuSiQue 8k locally.
cd /home/j0018/Desktop/tokenbank
run() {
  local bench=$1 budget=$2 cfgs=$3 tag=$4 log=$5; shift 5
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
  VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
  BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=32 SPLIT=test \
  V5=1 CONFIG_ONLY="$cfgs" OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
# smoke: 40 MATH items, all three arms -- confirms routing/escalation behave
run MATH 24000 "V5-Unconstrained,V5-Unc-Fmt,V5.1-NoPool" _q38smoke _q38_smoke.log MATH_N=40 SPLIT=full
echo "[$(date '+%F %T')] SMOKE DONE"
