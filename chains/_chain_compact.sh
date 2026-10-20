#!/bin/bash
# Compaction baseline on the four main benchmarks.
# The previous attempt passed arguments in the wrong order: run() takes
# (bench, budget, config, tag) and the calls omitted the config, so CONFIG_ONLY
# received the tag, matched nothing, and every step "completed" in 20 seconds
# having run zero tasks. The config name is explicit in every call below.
set -e
cd /home/j0018/Desktop/tokenbank
source ./_llm5_env.sh
run() {
  local bench=$1 budget=$2 cfg=$3 tag=$4; shift 4
  echo "[$(date '+%F %T')] START $tag  ($bench @$budget, cfg=$cfg)"
  env BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=$WORKERS SPLIT=test \
      V5=1 CONFIG_ONLY="$cfg" OUT_TAG=$tag TEMPERATURE=0 "$@" \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
C=V5-Compact-Real
for b in 16000 8000 24000; do
  k=$((b/1000))k
  run GAIA    $b $C _cmp_ga$k GAIA_N=1000
  run MATH    $b $C _cmp_ma$k MATH_N=500
  run FRAMES  $b $C _cmp_fr$k x=1
  run MUSIQUE $b $C _cmp_mu$k MUSIQUE_N=99999
done
echo "[$(date '+%F %T')] COMPACT ALL DONE"
