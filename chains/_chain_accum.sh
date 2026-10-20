#!/bin/bash
# Accumulate-only arm, to separate the two things the compaction baseline changes.
set -e
cd /home/j0018/Desktop/tokenbank
source ./_llm5_env.sh
echo "[$(date '+%F %T')] accum waiting for the compaction chain"
while pgrep -f 'chain_compact[.]sh' >/dev/null; do sleep 60; done
run() {
  local bench=$1 budget=$2 cfg=$3 tag=$4; shift 4
  echo "[$(date '+%F %T')] START $tag  ($bench @$budget, cfg=$cfg)"
  env BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=$WORKERS SPLIT=test \
      V5=1 CONFIG_ONLY="$cfg" OUT_TAG=$tag TEMPERATURE=0 "$@" \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
A=V5-Accum-Only
for b in 16000 8000 24000; do
  k=$((b/1000))k
  run GAIA    $b $A _acc_ga$k GAIA_N=1000
  run MATH    $b $A _acc_ma$k MATH_N=500
  run FRAMES  $b $A _acc_fr$k x=1
  run MUSIQUE $b $A _acc_mu$k MUSIQUE_N=99999
done
echo "[$(date '+%F %T')] ACCUM ALL DONE"
