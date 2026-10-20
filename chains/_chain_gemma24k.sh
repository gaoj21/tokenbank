#!/bin/bash
# wait for the GAIA-curve chain to finish, then run the 24K point on the
# other three datasets (300-task subsets, 3 configs each).
cd /home/j0018/Desktop/tokenbank
while pgrep -f "chain_gemma.sh" > /dev/null || pgrep -f "_budget_sweep_v2.py" > /dev/null; do sleep 120; done
run() {  # bench configs tag log [extra_env...]
  local bench=$1 configs=$2 tag=$3 log=$4; shift 4
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=gemma-4-31B-it \
  BENCHMARK=$bench BUDGETS=24000 SEEDS=42 MAX_WORKERS=16 CONFIG_ONLY="$configs" \
  OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
CFG="Unconstrained,Static Quota,Improved+Compact"
run FRAMES  "$CFG" _fr24k_gem _gemma_frames24k.log FRAMES_N=300
run MATH    "$CFG" _ma24k_gem _gemma_math24k.log  MATH_N=300
run MUSIQUE "$CFG" _mu24k_gem _gemma_musique24k.log MUSIQUE_N=300
