#!/bin/bash
# Rebuild the main table with the verbalized-escalation channel REMOVED.
#
# Ab7 is the full system minus m2_verbal, so the ladder is routed by the
# log-probability gate alone. The channel was introduced because that gate
# backfired on Qwen3.6; on Qwen3.8 it does not. Measured at 24k, dropping the
# channel gains +3.8pp on MATH (p=0.024) at 47% fewer tokens and ties exactly
# on MuSiQue (55/55, p=1.000) at 16% fewer tokens. It is not ahead anywhere, so
# it does not earn a place in the method and the table should be built without it.
#
# MATH@24k and MuSiQue@24k already exist and are not re-run.
set -e
cd /home/j0018/Desktop/tokenbank
run() {
  local bench=$1 budget=$2 tag=$3 extra=$4
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
    VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
    BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=32 SPLIT=test \
    V5=1 CONFIG_ONLY="V5.1-Ab7-NoVerbal" OUT_TAG=$tag TEMPERATURE=0 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
# cheapest first so the table fills progressively
run GAIA    24000 _q38ab7_ga24 GAIA_N=1000
run GAIA    16000 _q38ab7_ga16 GAIA_N=1000
run GAIA     8000 _q38ab7_ga8  GAIA_N=1000
run MATH    16000 _q38ab7_ma16 MATH_N=500
run MATH     8000 _q38ab7_ma8  MATH_N=500
run FRAMES  24000 _q38ab7_fr24 x=1
run FRAMES  16000 _q38ab7_fr16 x=1
run FRAMES   8000 _q38ab7_fr8  x=1
run MUSIQUE 16000 _q38ab7_mu16 MUSIQUE_N=99999
run MUSIQUE  8000 _q38ab7_mu8  MUSIQUE_N=99999
echo "[$(date '+%F %T')] AB7 FULL DONE"
