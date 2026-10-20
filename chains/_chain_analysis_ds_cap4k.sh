#!/bin/bash
# Follow-up: per-call generation cap raised to 4096. In the all-escalate run, L2 executor
# calls used the default MAX_TOKENS=1024 with thinking on, and 84% of GAIA L2 answers came
# back empty; every GAIA item L2 "harmed" had hit the cap. These runs ask whether the loop
# helps once it is not truncated. Waits for the first chain to finish.
cd /home/j0018/Desktop/tokenbank
while ! grep -q "DS ANALYSIS ALL DONE" logs/_chain_analysis_ds.log 2>/dev/null; do sleep 60; done
TAG=DeepSeek-V4.1-Flash
base() { env OPENAI_BASE_URL=http://127.0.0.1:18000/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=deepseek-v4.1-flash \
  MODEL_TAG=$TAG SEEDS=42 MAX_WORKERS=256 TEMPERATURE=0 V5=1 L01_THINKING_OFF=1 MAX_TOKENS=4096 "$@"; }
run() { local t=$1; shift
  [ -s "traces/tasks_sweep_$TAG$t.jsonl" ] && { echo "SKIP $t"; return; }
  echo "[$(date '+%F %T')] START $t"; base OUT_TAG=$t "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
for B in MATH:ma:MATH_N=500 GAIA:ga:GAIA_N=1000 FRAMES:fr:x=1 MUSIQUE:mu:MUSIQUE_N=99999; do
  IFS=: read bench t extra <<< "$B"
  run _dsx4k_${t}24 BENCHMARK=$bench BUDGETS=24000 SPLIT=test $extra CONFIG_ONLY=V5-Unc-Fmt,V5.2-ReasonLadder-AllEsc
done
echo "[$(date '+%F %T')] DS CAP4K ALL DONE"
