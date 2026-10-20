#!/bin/bash
# Qwen3.8 becomes the main model, so the two analyses that are model-specific
# have to be rebuilt on it:
#   1. The additive attribution of the MATH gain (paper Table 2). Requires the
#      ladder DISABLED, otherwise every single-mechanism ablation reads null
#      because L0 routing parks most items in a single call. Ab6 = M3ctx+M5 only,
#      Ab3 = +M1, NoPool = +M2 (already have), Unc-Fmt = baseline (already have).
#   2. The ladder-on "shadow" ablations (paper Table 4), which exist to show the
#      bypass artifact. Ab1 = -M1, Ab5 = -M3ctx.
# Then the run-to-run noise floor, re-running identical configs at temperature 0.
set -e
cd /home/j0018/Desktop/tokenbank
run() {
  local bench=$1 budget=$2 cfgs=$3 tag=$4 log=$5; shift 5
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
    VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
    BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=24 SPLIT=test \
    V5=1 CONFIG_ONLY="$cfgs" OUT_TAG=$tag TEMPERATURE=0 "$@" \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
run MATH 24000 "V5.1-Ab6-NoM2NoM1,V5.1-Ab3-NoM2,V5.1-Ab1-NoM1,V5.1-Ab5-NoM3ctx" \
    _q38_abl_ma _q38_abl_ma.log MATH_N=500
run MATH 24000 "V5-Unc-Fmt,V5.1-NoPool" _q38_rr_ma _q38_rr_ma.log MATH_N=500
run GAIA 24000 "V5-Unc-Fmt,V5.1-NoPool" _q38_rr_ga _q38_rr_ga.log GAIA_N=1000
echo "[$(date '+%F %T')] ATTRIB ALL DONE"
