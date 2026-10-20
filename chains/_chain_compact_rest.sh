#!/bin/bash
# Remaining compaction cells, queued behind the SWE regeneration. Running the two
# in parallel halved throughput: the server saturates around 32-40 concurrent
# requests, so two jobs at 26 tasks/min together are slower than one at 95.
set -e
cd /home/j0018/Desktop/tokenbank
source ./_llm5_env.sh
echo "[$(date '+%F %T')] waiting for the SWE regeneration"
while pgrep -f 'chain_swe_v2[.]sh' >/dev/null; do sleep 60; done
run() {
  local bench=$1 budget=$2 tag=$3; shift 3
  echo "[$(date '+%F %T')] START $tag"
  env BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=$WORKERS SPLIT=test \
      V5=1 CONFIG_ONLY="V5-Compact-Real" OUT_TAG=$tag TEMPERATURE=0 "$@" \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag  (API errors: $(grep -c '\[APIERR\]' logs/$tag.log))"
}
run MUSIQUE 24000 _cmp3_mu24k MUSIQUE_N=99999
for spec in "GAIA _cmp3_ga8k GAIA_N=1000" "MATH _cmp3_ma8k MATH_N=500" \
            "FRAMES _cmp3_fr8k x=1" "MUSIQUE _cmp3_mu8k MUSIQUE_N=99999"; do
  set -- $spec; run "$1" 8000 "$2" "$3"
done
echo "[$(date '+%F %T')] COMPACT3 REST DONE"
