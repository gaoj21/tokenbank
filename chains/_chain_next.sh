#!/bin/bash
# Everything still owed, on llm5 at 64 concurrency (measured 95 tasks/min once
# the endpoint is not contended; the earlier 25/min reading was congestion).
# Serial: the server saturates below the sum of several jobs' concurrency.
set -e
cd /home/j0018/Desktop/tokenbank
source ./_llm5_env.sh
echo "[$(date '+%F %T')] waiting for the speed/consistency run"
while pgrep -f 'budget_sweep_v2[.]py' >/dev/null; do sleep 20; done

run() {
  local bench=$1 budget=$2 cfg=$3 tag=$4; shift 4
  echo "[$(date '+%F %T')] START $tag"
  env BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=$WORKERS SPLIT=${SPLIT:-test} \
      V5=1 CONFIG_ONLY="$cfg" OUT_TAG=$tag TEMPERATURE=0 "$@" \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}

# 1. the SWE arm the local evaluation is waiting on
SPLIT=full run SWEO 24000 "V5.1-Ab7-NoVerbal" _sweo_ab7 SWE_N=500 V5_MAX_ROUNDS=20 DUMP_RAW_ANSWER=1
echo "[$(date '+%F %T')] --- SWE arm done ---"

# 2. BigCodeBench 8k and 16k; 4k already ran. 16k is the non-binding control
#    where the three arms should converge if the gains really come from the cap.
for b in 8000 16000; do
  SPLIT=full run BCB $b "BCB-Naive,BCB-Ours,BCB-Ab7" _bcb$((b/1000))k BCB_N=500 V5_MAX_ROUNDS=5 DUMP_RAW_ANSWER=1
done
echo "[$(date '+%F %T')] --- BigCodeBench done ---"

# 3. compaction on the four main benchmarks, all three budgets
for b in 16000 8000 24000; do
  k=$((b/1000))k
  run GAIA    $b _cmp_ga$k GAIA_N=1000
  run MATH    $b _cmp_ma$k MATH_N=500
  run FRAMES  $b _cmp_fr$k x=1
  run MUSIQUE $b _cmp_mu$k MUSIQUE_N=99999
done
echo "[$(date '+%F %T')] NEXT ALL DONE"
