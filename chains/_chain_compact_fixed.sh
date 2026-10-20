#!/bin/bash
# Compaction baseline, third attempt. The first used a spend-based trigger; the
# second built a message list Qwen's template rejects, so every call after a
# summarization returned 400 and every compacted task scored zero. Both are
# fixed. API errors are now logged, so a repeat of this is visible.
set -e
cd /home/j0018/Desktop/tokenbank
source ./_llm5_env.sh
run() {
  local bench=$1 budget=$2 cfg=$3 tag=$4; shift 4
  echo "[$(date '+%F %T')] START $tag  ($bench @$budget)"
  env BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=$WORKERS SPLIT=test \
      V5=1 CONFIG_ONLY="$cfg" OUT_TAG=$tag TEMPERATURE=0 "$@" \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  local err=$(grep -c "\[APIERR\]" logs/$tag.log 2>/dev/null)
  echo "[$(date '+%F %T')] DONE  $tag   (API errors: ${err:-0})"
}
C=V5-Compact-Real
for b in 16000 24000 8000; do
  k=$((b/1000))k
  run GAIA    $b $C _cmp3_ga$k GAIA_N=1000
  run MATH    $b $C _cmp3_ma$k MATH_N=500
  run FRAMES  $b $C _cmp3_fr$k x=1
  run MUSIQUE $b $C _cmp3_mu$k MUSIQUE_N=99999
done
echo "[$(date '+%F %T')] COMPACT3 ALL DONE"
